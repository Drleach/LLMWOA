import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import os
import time
from opfunu.cec_based.cec2022 import *
from openpyxl import Workbook
from openpyxl.styles import Font
from openpyxl.utils.dataframe import dataframe_to_rows

# 设置中文字体支持
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# DeepSeek模型封装 * deepseek-chat 和 deepseek-reasoner 都已经升级为 DeepSeek-V3.1。
#deepseek-chat 对应 DeepSeek-V3.1 的非思考模式，
#deepseek-reasoner 对应 DeepSeek-V3.1 的思考模式.
class DeepSeekModel:
    def __init__(self, api_key: str, model_name: str = "deepseek-reasoner"):
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://api.deepseek.com/v1"
            )
            self.model = model_name
        except ImportError:
            raise ImportError("请安装openai库: pip install openai")
        except Exception as e:
            raise Exception(f"初始化DeepSeek模型失败: {str(e)}")

    def generate_content(self, prompt: str, **kwargs):
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": kwargs.get("system_prompt", "You are a helpful assistant.")},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_output_tokens", 2048)
            )
            # 构建响应对象
            class Response:
                def __init__(self, text, raw):
                    self.text = text
                    self.raw = raw
            return Response(response.choices[0].message.content, response)
        except Exception as e:
            print(f"DeepSeek API调用失败: {str(e)}")
            # 返回一个默认响应，避免程序崩溃
            class DefaultResponse:
                def __init__(self):
                    self.text = "a=1.0\nb=1.0\n探索概率=0.5，开发概率=0.5"
                    self.raw = None
            return DefaultResponse()

# LLM WOA 顾问
class LLM_WOA_Advisor:
    def __init__(self, llm_model):
        self.llm = llm_model  # LLM模型（DeepSeek）
    
    def get_woa_strategy(self, state):
        """根据当前状态，让LLM生成WOA的参数和策略调整建议"""
        prompt = f"""你是鲸鱼优化算法（WOA）的智能调优顾问，需要根据当前优化状态调整参数和策略。
        WOA核心参数说明：
        - 收缩因子a：控制探索（a>1）与开发（a<1），传统线性下降（从2→0）
        - 螺旋系数b：控制螺旋更新的强度（通常固定为1）
        - 搜索策略：包围捕食（局部开发）、螺旋更新（平衡）、随机搜索（全局探索）
        
        优化目标：在保持种群多样性的同时加速收敛，避免局部最优。
        
        当前状态：
        {state}
        
        请输出：
        1. 建议的a值（格式：a=具体数值，范围0~2）
        2. 建议的b值（格式：b=具体数值，范围0.5~2）
        3. 建议的策略概率（格式：探索概率=P1，开发概率=P2，其中P1+P2=1）
        输出示例：
        a=1.2
        b=1.0
        探索概率=0.6，开发概率=0.4
        """
        
        # 调用LLM获取建议
        response = self.llm.generate_content(prompt)
        return self._parse_response(response.text)
    
    def _parse_response(self, text):
        """解析LLM输出，提取参数和策略"""
        try:
            a = float(re.findall(r'a=([\d.]+)', text)[0])
            b = float(re.findall(r'b=([\d.]+)', text)[0])
            explore_p = float(re.findall(r'探索概率=([\d.]+)', text)[0])
            exploit_p = float(re.findall(r'开发概率=([\d.]+)', text)[0])
            return {
                'a': np.clip(a, 0, 2),
                'b': np.clip(b, 0.5, 2),
                'explore_prob': np.clip(explore_p, 0, 1),
                'exploit_prob': np.clip(exploit_p, 0, 1)
            }
        except:
            # 解析失败时返回默认值
            return {'a': 1.0, 'b': 1.0, 'explore_prob': 0.5, 'exploit_prob': 0.5}
    
    def get_escape_strategy(self, current_best):
        """当算法陷入局部最优时，获取LLM的跳出策略建议"""
        prompt = f"""当前WOA陷入局部最优，最优适应度连续多代无明显改进（当前值：{current_best:.6f}）。
        请生成一个种群扰动策略，帮助算法跳出局部最优。
        输出格式：
        扰动比例=P（0~0.3，需要扰动的个体比例）
        噪声强度=S（0~0.2，相对于搜索空间范围的噪声比例）
        示例：
        扰动比例=0.2
        噪声强度=0.05
        """
        
        response = self.llm.generate_content(prompt)
        return self._parse_escape_response(response.text)
    
    def _parse_escape_response(self, text):
        """解析跳出策略的响应"""
        try:
            perturb_ratio = float(re.findall(r'扰动比例=([\d.]+)', text)[0])
            noise_strength = float(re.findall(r'噪声强度=([\d.]+)', text)[0])
            return {
                'perturb_ratio': np.clip(perturb_ratio, 0, 0.3),
                'noise_strength': np.clip(noise_strength, 0, 0.2)
            }
        except:
            # 解析失败时返回默认值
            return {'perturb_ratio': 0.2, 'noise_strength': 0.05}

# 辅助函数：获取优化状态
def get_optimization_state(pop, fitness, historical_best, iter, max_iter):
    """提取WOA当前优化状态，转化为LLM可理解的文本描述"""
    # 计算种群多样性（个体与最优解的平均距离）
    best_idx = np.argmin(fitness)
    diversity = np.mean([np.linalg.norm(ind - pop[best_idx]) for ind in pop])
    
    # 计算近期改进量（最近5代）
    recent_improve = np.mean(np.diff(historical_best[-5:])) if len(historical_best)>=5 else 0
    
    state = f"""当前优化状态：
    - 迭代进度：{iter}/{max_iter}（{iter/max_iter*100:.1f}%）
    - 最优适应度：{np.min(fitness):.6f}
    - 种群多样性：{diversity:.6f}（值越大多样性越高）
    - 近期改进量：{recent_improve:.6f}（值越大改进越快）
    - 收敛状态：{'初期探索' if iter/max_iter < 0.3 else '中期开发' if iter/max_iter < 0.7 else '后期收敛'}
    """
    return state

# 辅助函数：检查是否陷入局部最优
def check_stuck(historical_best, patience=10, epsilon=1e-6):
    """判断是否陷入局部最优（连续patience代无显著改进）"""
    if len(historical_best) < patience:
        return False
    recent = historical_best[-patience:]
    improvement = recent[0] - recent[-1]
    return improvement < epsilon  # 改进量小于阈值，判定为停滞

# 传统WOA算法
def traditional_woa(func, dim, pop_size=30, max_iter=100):
    """传统鲸鱼优化算法"""
    # 初始化种群
    lb, ub = func.lb, func.ub
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([func.evaluate(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best_solution = pop[best_idx].copy()
    best_fitness = [fitness[best_idx]]  # 记录历史最优
    
    for iter in range(max_iter):
        # 传统WOA参数：a线性下降从2到0
        a = 2 - iter * (2 / max_iter)
        b = 1  # 固定螺旋系数
        
        for i in range(pop_size):
            # 随机选择一个个体作为目标
            r1 = np.random.rand()  # [0,1)随机数
            r2 = np.random.rand()  # [0,1)随机数
            
            A = 2 * a * r1 - a  # 系数向量
            C = 2 * r2          # 系数向量
            
            p = np.random.rand()
            if p < 0.5:
                if np.abs(A) >= 1:
                    # 随机搜索
                    rand_idx = np.random.randint(pop_size)
                    X_rand = pop[rand_idx]
                    D = np.abs(C * X_rand - pop[i])
                    pop[i] = X_rand - A * D
                else:
                    # 包围捕食
                    D = np.abs(C * best_solution - pop[i])
                    pop[i] = best_solution - A * D
            else:
                # 螺旋更新
                D = np.abs(best_solution - pop[i])
                l = (np.random.rand() - 1) * 2  # [-1,1)随机数
                pop[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_solution
            
            # 边界处理
            pop[i] = np.clip(pop[i], lb, ub)
        
        # 评估新种群并更新最优解
        fitness = np.array([func.evaluate(ind) for ind in pop])
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness[-1]:
            best_solution = pop[current_best_idx].copy()
        best_fitness.append(np.min(fitness))
    
    return best_solution, best_fitness

# LLM-WOA算法
def llm_woa(func, dim, pop_size=30, max_iter=100, llm_advisor=None):
    """结合LLM的改进WOA算法"""
    if llm_advisor is None:
        raise ValueError("必须提供LLM顾问实例")
    
    # 初始化种群
    lb, ub = func.lb, func.ub
    pop = np.random.uniform(lb, ub, (pop_size, dim))
    fitness = np.array([func.evaluate(ind) for ind in pop])
    best_idx = np.argmin(fitness)
    best_solution = pop[best_idx].copy()
    best_fitness = [fitness[best_idx]]  # 记录历史最优
    
    # LLM决策频率（每5代调用一次，平衡效率与性能）
    llm_call_interval = 5
    llm_strategy = {'a': 2.0, 'b': 1.0, 'explore_prob': 0.5, 'exploit_prob': 0.5}
    
    for iter in range(max_iter):
        # 每间隔一定迭代，调用LLM更新策略
        if iter % llm_call_interval == 0:
            state = get_optimization_state(pop, fitness, best_fitness, iter, max_iter)
            llm_strategy = llm_advisor.get_woa_strategy(state)
        
        a = llm_strategy['a']  # 由LLM动态调整的收缩因子
        b = llm_strategy['b']  # 由LLM动态调整的螺旋系数
        explore_prob = llm_strategy['explore_prob']
        
        # 检查是否陷入局部最优，如果是则调用LLM获取跳出策略
        if check_stuck(best_fitness):
            escape_strategy = llm_advisor.get_escape_strategy(best_fitness[-1])
            # 执行扰动
            perturb_ratio = escape_strategy['perturb_ratio']
            noise_strength = escape_strategy['noise_strength']
            perturb_idx = np.random.choice(pop_size, int(pop_size*perturb_ratio), replace=False)
            pop[perturb_idx] += noise_strength * (ub - lb) * np.random.normal(0, 1, (len(perturb_idx), dim))
            pop[perturb_idx] = np.clip(pop[perturb_idx], lb, ub)
        
        for i in range(pop_size):
            # 随机选择一个个体作为目标
            r1 = np.random.rand()  # [0,1)随机数
            r2 = np.random.rand()  # [0,1)随机数
            
            A = 2 * a * r1 - a  # 系数向量
            C = 2 * r2          # 系数向量
            
            # 根据LLM建议的概率选择探索/开发策略
            if np.random.rand() < explore_prob:
                # 探索策略：随机搜索（全局勘探）
                if np.abs(A) >= 1:
                    rand_idx = np.random.randint(pop_size)
                    X_rand = pop[rand_idx]
                    D = np.abs(C * X_rand - pop[i])
                    pop[i] = X_rand - A * D
                else:
                    # 包围捕食（局部开发）
                    D = np.abs(C * best_solution - pop[i])
                    pop[i] = best_solution - A * D
            else:
                # 开发策略：螺旋更新（精细搜索）
                D = np.abs(best_solution - pop[i])
                l = (np.random.rand() - 1) * 2  # [-1,1)随机数
                pop[i] = D * np.exp(b * l) * np.cos(2 * np.pi * l) + best_solution
            
            # 边界处理
            pop[i] = np.clip(pop[i], lb, ub)
        
        # 评估新种群并更新最优解
        fitness = np.array([func.evaluate(ind) for ind in pop])
        current_best_idx = np.argmin(fitness)
        if fitness[current_best_idx] < best_fitness[-1]:
            best_solution = pop[current_best_idx].copy()
        best_fitness.append(np.min(fitness))
    
    return best_solution, best_fitness

# 运行实验并比较结果
def run_experiment(dim=10, pop_size=30, max_iter=100, trials=5, save_dir="./results"):
    """运行传统WOA与LLM-WOA的对比实验"""
    # 创建保存结果的目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(os.path.join(save_dir, "convergence_plots")):
        os.makedirs(os.path.join(save_dir, "convergence_plots"))
    
    # 初始化DeepSeek模型和LLM顾问
    api_key = "这里输入你的API"
    deepseek_model = DeepSeekModel(api_key=api_key)
    llm_advisor = LLM_WOA_Advisor(deepseek_model)
    
    # 选择CEC2022测试函数（这里选择前5个作为示例，可根据需要扩展）
    cec_functions = [
        F12022(dim), F22022(dim), F32022(dim),
        F42022(dim), F52022(dim)
    ]
    func_names = [f"F{i+1}2022" for i in range(len(cec_functions))]
    
    # 存储结果的列表
    results = []
    
    for func_idx, func in enumerate(cec_functions):
        func_name = func_names[func_idx]
        print(f"正在运行 {func_name} (维度: {dim})...")
        
        # 存储多次试验的结果
        traditional_results = []
        llm_woa_results = []
        traditional_traces = []
        llm_woa_traces = []
        
        for trial in range(trials):
            print(f"  试验 {trial+1}/{trials}")
            
            # 设置随机种子，保证对比公平性
            np.random.seed(42 + trial)
            
            # 运行传统WOA
            start_time = time.time()
            best_sol_traditional, trace_traditional = traditional_woa(
                func, dim, pop_size, max_iter
            )
            traditional_time = time.time() - start_time
            traditional_best = np.min(trace_traditional)
            traditional_results.append({
                'best': traditional_best,
                'time': traditional_time
            })
            traditional_traces.append(trace_traditional)
            
            # 运行LLM-WOA
            start_time = time.time()
            best_sol_llm, trace_llm = llm_woa(
                func, dim, pop_size, max_iter, llm_advisor
            )
            llm_time = time.time() - start_time
            llm_best = np.min(trace_llm)
            llm_woa_results.append({
                'best': llm_best,
                'time': llm_time
            })
            llm_woa_traces.append(trace_llm)
        
        # 计算统计结果
        traditional_best_vals = [res['best'] for res in traditional_results]
        traditional_times = [res['time'] for res in traditional_results]
        
        llm_best_vals = [res['best'] for res in llm_woa_results]
        llm_times = [res['time'] for res in llm_woa_results]
        
        # 保存结果
        results.append({
            'function': func_name,
            'traditional_best': np.min(traditional_best_vals),
            'traditional_mean': np.mean(traditional_best_vals),
            'traditional_std': np.std(traditional_best_vals),
            'traditional_time': np.mean(traditional_times),
            'llm_woa_best': np.min(llm_best_vals),
            'llm_woa_mean': np.mean(llm_best_vals),
            'llm_woa_std': np.std(llm_best_vals),
            'llm_woa_time': np.mean(llm_times)
        })
        
        # 绘制收敛曲线
        plt.figure(figsize=(10, 6))
        
        # 传统WOA平均收敛曲线
        traditional_mean_trace = np.mean(traditional_traces, axis=0)
        plt.plot(traditional_mean_trace, 'b-', linewidth=2, label='传统WOA')
        
        # LLM-WOA平均收敛曲线
        llm_mean_trace = np.mean(llm_woa_traces, axis=0)
        plt.plot(llm_mean_trace, 'r-', linewidth=2, label='LLM-WOA')
        
        plt.xlabel('迭代次数')
        plt.ylabel('最优适应度值')
        plt.title(f'{func_name} 收敛曲线 (维度: {dim})')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        plt.yscale('log')  # 使用对数尺度更适合展示优化过程
        plt.tight_layout()
        
        # 保存收敛曲线
        plot_path = os.path.join(save_dir, "convergence_plots", f"{func_name}_dim{dim}.png")
        plt.savefig(plot_path, dpi=300)
        plt.close()
        print(f"  收敛曲线已保存至: {plot_path}")
    
    # 将结果保存到Excel
    df = pd.DataFrame(results)
    excel_path = os.path.join(save_dir, f"woa_vs_llm_woa_dim{dim}.xlsx")
    
    # 创建Excel工作簿并设置格式
    wb = Workbook()
    ws = wb.active
    ws.title = f"维度_{dim}"
    
    # 添加数据
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)
    
    # 设置表头样式
    header_font = Font(bold=True)
    for cell in ws[1]:
        cell.font = header_font
    
    # 对优于传统WOA的数值进行加黑处理
    bold_font = Font(bold=True)
    # 从第二行开始（跳过表头），到最后一行
    for row in range(2, ws.max_row + 1):
        # 比较最优值（值越小越好）
        if ws.cell(row=row, column=6).value < ws.cell(row=row, column=2).value:
            ws.cell(row=row, column=6).font = bold_font
        
        # 比较平均值（值越小越好）
        if ws.cell(row=row, column=7).value < ws.cell(row=row, column=3).value:
            ws.cell(row=row, column=7).font = bold_font
        
        # 比较标准差（值越小越好）
        if ws.cell(row=row, column=8).value < ws.cell(row=row, column=4).value:
            ws.cell(row=row, column=8).font = bold_font
    
    # 保存Excel文件
    wb.save(excel_path)
    print(f"实验结果已保存至: {excel_path}")
    
    return df

if __name__ == "__main__":
    # 运行实验
    # 可以调整参数：维度(dim)、种群大小(pop_size)、最大迭代次数(max_iter)、试验次数(trials)
    run_experiment(
        dim=10,          # 问题维度
        pop_size=30,     # 种群大小
        max_iter=100,    # 最大迭代次数
        trials=5,        # 每次测试函数运行的试验次数
        save_dir="./woa_comparison_results"  # 结果保存目录
    )
    print("所有实验完成！")
