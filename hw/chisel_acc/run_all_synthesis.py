import os
import subprocess
import csv
import glob
import shutil
import re  # <--- 必须导入这个模块

# --- 配置 ---
OPENLANE_CONTAINER = "snax_cluster_devcontainer-openlane-1"
BASE_GENERATED_PATH = "generated/dot_product"
WORKING_DIR = "/workspaces/snax_cluster/hw/chisel_acc"
SUMMARY_FILE = "synthesis_results_all.csv"
RUNS_DIR = os.path.join(WORKING_DIR, "runs")

def create_config(design_name, verilog_file):
    config_content = f"""
set ::env(DESIGN_NAME) "{design_name}"
set ::env(VERILOG_FILES) [glob ./{verilog_file}]
set ::env(CLOCK_PORT) "clock"
set ::env(CLOCK_PERIOD) "25.0"
set ::env(FP_CORE_UTIL) 40
set ::env(RT_MAX_LAYER) "met4"
set ::env(FP_PDN_MULTILAYER) 0
# 强制保留层级
set ::env(SYNTH_FLAT) 0
set ::env(SYNTH_HIERARCHICAL) 1
set ::env(SYNTH_NO_FLAT) 1
"""
    with open("config.tcl", "w") as f:
        f.write(config_content)

def parse_metrics(run_dir):
    area, slack, power = "N/A", "N/A", "N/A"
    csv_path = os.path.join(run_dir, "reports/metrics.csv")
    if os.path.exists(csv_path):
        try:
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    area = row.get("DieArea_um^2", "N/A")
                    slack = row.get("wns", "N/A")
                    power = row.get("total_power_W", "N/A")
        except: pass
    return area, slack, power

def parse_hierarchy_details(log_path):
    """从合成日志中提取所有子模块的面积明细"""
    details = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            content = f.read()
            # 匹配 Yosys 的面积报告行
            matches = re.findall(r"Chip area for module '(.+?)':\s+([0-9.]+)", content)
            for mod_name, mod_area in matches:
                clean_name = mod_name.lstrip('\\')
                details.append(f"{clean_name}:{mod_area}")
    return " | ".join(details) if details else "Flattened"

def run_batch():
    # 初始化汇总表（增加 Hierarchy 列）
    if not os.path.exists(SUMMARY_FILE):
        with open(SUMMARY_FILE, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Design_Name", "Total_Area(um^2)", "Slack(ns)", "Power(W)", "Hierarchy_Details"])

    sv_files = []
    for root, dirs, files in os.walk(BASE_GENERATED_PATH):
        for file in files:
            if file.endswith(".sv"):
                sv_files.append(os.path.join(root, file))

    print(f"找到 {len(sv_files)} 个设计。即时清理模式已开启。")

    for sv_path in sv_files:
        design_name = os.path.basename(sv_path).replace(".sv", "")
        v_file = f"{design_name}_converted.v"
        v_path = os.path.join(WORKING_DIR, v_file)
        
        print(f"\n🚀 正在处理: {design_name}")

        try:
            # 1. 转换
            subprocess.run(f"sv2v --top {design_name} {sv_path} > {v_path}", shell=True, check=True)

            # 2. 配置
            create_config(design_name, v_file)

            # 3. 运行 (即使 Flow 报 Setup Error 也会继续，因为我们需要面积数据)
            docker_cmd = f"docker exec -u 0 -w {WORKING_DIR} {OPENLANE_CONTAINER} flow.tcl -design . -pdk sky130A -overwrite"
            subprocess.run(docker_cmd, shell=True)

            # 4. 提取数据
            run_dirs = sorted(glob.glob(os.path.join(RUNS_DIR, "RUN_*")), reverse=True)
            if run_dirs:
                latest_run = run_dirs[0]
                total_area, slack, power = parse_metrics(latest_run)
                
                # 提取子模块明细
                log_file = os.path.join(latest_run, "logs/synthesis/1-synthesis.log")
                hierarchy_details = parse_hierarchy_details(log_file)
                
                # 写入 CSV
                with open(SUMMARY_FILE, "a", newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow([design_name, total_area, slack, power, hierarchy_details])
                
                print(f"📈 提取结果: Area={total_area}, Hierarchy={hierarchy_details}")
                
                # 5. 清理空间
                shutil.rmtree(latest_run)
                if os.path.exists(v_path): os.remove(v_path)
                print(f"🧹 已清理 RUN 目录及临时文件。")
            else:
                print(f"⚠️ 未找到运行目录。")

        except Exception as e:
            print(f"❌ 严重错误: {e}")

    print(f"\n✨ 全部完成！汇总表：{SUMMARY_FILE}")

if __name__ == "__main__":
    run_batch()