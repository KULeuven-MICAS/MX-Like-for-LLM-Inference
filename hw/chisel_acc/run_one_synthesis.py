import os
import subprocess
import csv
import glob
import re

# --- 配置：修改为你当前想测试的文件 ---
TEST_SV_FILE = "generated/dot_product/E2M1_E2M1_UE2M6/DotProductUnit_E2M1_x_E2M1_scale_UE2M6.sv"
OPENLANE_CONTAINER = "snax_cluster_devcontainer-openlane-1"
WORKING_DIR = "/workspaces/snax_cluster/hw/chisel_acc"

def create_config(design_name, verilog_file):
    config_content = f"""
set ::env(DESIGN_NAME) "{design_name}"
set ::env(VERILOG_FILES) [glob ./{verilog_file}]
set ::env(CLOCK_PORT) "clock"
set ::env(CLOCK_PERIOD) "20.0"
set ::env(FP_CORE_UTIL) 40
set ::env(RT_MAX_LAYER) "met4"
set ::env(FP_PDN_MULTILAYER) 0

# --- 尝试保留层级 ---
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
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                area = row.get("DieArea_um^2", "N/A")
                slack = row.get("wns", "N/A")
                power = row.get("total_power_W", "N/A")
    return area, slack, power

def parse_hierarchy(log_path):
    hierarchy = []
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            content = f.read()
            # 匹配 Yosys 打印的模块面积
            matches = re.findall(r"Chip area for module '(.+?)':\s+([0-9.]+)", content)
            for mod, area in matches:
                hierarchy.append((mod.lstrip('\\'), area))
    return hierarchy

def run_test():
   # 获取不带后缀的文件名，例如 "DotProductUnit_E2M1_x_E2M1_scale_UE2M6"
    design_name = os.path.basename(TEST_SV_FILE).replace(".sv", "")
    sv_path = os.path.abspath(TEST_SV_FILE)
    v_file = f"{design_name}_converted.v"
    v_path = os.path.join(WORKING_DIR, v_file)

    print(f"--- 步骤 1: 转换 (使用 --top {design_name}) ---")
    # 正确写法：--top 后面跟模块名，sv_path 放在最后作为输入文件
    subprocess.run(f"sv2v --top {design_name} {sv_path} > {v_path}", shell=True)

    print(f"--- 步骤 2: 生成配置 ---")
    create_config(design_name, v_file)

    print(f"--- 步骤 3: 运行 OpenLane ---")
    docker_cmd = f"docker exec -u 0 -w {WORKING_DIR} {OPENLANE_CONTAINER} flow.tcl -design . -pdk sky130A -overwrite"
    subprocess.run(docker_cmd, shell=True)

    print(f"\n" + "="*50)
    print(f"综合结果汇总 - {design_name}")
    print("="*50)

    latest_run = sorted(glob.glob(os.path.join(WORKING_DIR, "runs/RUN_*")), reverse=True)[0]
    area, slack, power = parse_metrics(latest_run)
    
    print(f"总面积 (Total Area): {area} um^2")
    print(f"时序余量 (Worst Slack): {slack} ns")
    print(f"预估功耗 (Total Power): {power} W")
    
    print("-" * 30)
    print("层级拆解 (Hierarchy Analysis):")
    log_file = os.path.join(latest_run, "logs/synthesis/1-synthesis.log")
    modules = parse_hierarchy(log_file)
    
    if not modules:
        print("  [警告] 未发现子模块，可能已被 sv2v 或 Yosys 拍平 (Flattened)。")
    else:
        for mod, mod_area in modules:
            print(f"  └─ {mod:<30} : {mod_area} um^2")
    print("="*50)

if __name__ == "__main__":
    run_test()