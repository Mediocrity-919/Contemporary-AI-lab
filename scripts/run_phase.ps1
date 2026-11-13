<#
分阶段运行实验（plan.md 第 8 节）。在 lab3 目录下执行，例如：

    powershell -ExecutionPolicy Bypass -File scripts\run_phase.ps1 1
    powershell -ExecutionPolicy Bypass -File scripts\run_phase.ps1 status

可以随时 Ctrl+C 中断；重新执行同一阶段会跳过已完成的运行，未完成的运行从中断那一轮之后继续。
每个阶段结束时，规则化的决策（学习率、增强档、是否加 SE / 标签平滑）会自动写入 configs/decisions.yaml，
依据记录在 configs/decisions_log.md。阶段 1 结束后需要人工确定训练轮数（见结尾提示）。
输出同时保存到 outputs/logs/。
#>
param([Parameter(Mandatory = $true)][string]$Phase)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = if ($env:LAB3_PYTHON) { $env:LAB3_PYTHON } else { "D:\Anaconda_envs\envs\pytorch_env\python.exe" }
$env:PYTHONIOENCODING = "utf-8"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8   # 让 PowerShell 按 UTF-8 解读 Python 的中文输出
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONUNBUFFERED = "1"
New-Item -ItemType Directory -Force "outputs\logs" | Out-Null
$Log = "outputs\logs\phase$Phase-$(Get-Date -Format yyyyMMdd-HHmmss).log"

function Invoke-Py {
    param([Parameter(ValueFromRemainingArguments = $true)][string[]]$CmdArgs)
    Write-Host "`n>>> python $($CmdArgs -join ' ')" -ForegroundColor Cyan
    # 不用 Tee-Object：PowerShell 5.1 下它写出的是 UTF-16，其他工具不便搜索；这里逐行追加为 UTF-8
    & $Py @CmdArgs | ForEach-Object { $_; Add-Content -LiteralPath $Log -Value $_ -Encoding UTF8 }
    if ($LASTEXITCODE -ne 0) { throw "命令失败（返回码 $LASTEXITCODE）：python $($CmdArgs -join ' ')" }
}

function Run { Invoke-Py -m src.run_experiments @args }
function Decide { Invoke-Py -m src.select_decisions @args }

# 依次做多个决策：某个决策失败（例如最优值在网格边缘、需要补点）时先记下，继续做其余决策，
# 最后统一报告并停止。这样一次就能看到所有需要补点的网格，不必来回重跑。
function Decide-All {
    param([string[][]]$Jobs)
    $failed = @()
    foreach ($job in $Jobs) {
        try { Decide @job } catch { $failed += ($job -join ' '); Write-Host $_ -ForegroundColor Red }
    }
    if ($failed.Count -gt 0) {
        throw ("以下决策未完成（见上方说明），处理后重跑本阶段：`n  " + ($failed -join "`n  "))
    }
}

$start = Get-Date
switch ($Phase) {
    "status" { Run "configs/**/*.yaml" --status }

    "1" {
        # 预实验（约 1.5 小时）：OurCNN 100 轮（约 25 分钟）+ 预训练微调 20 轮（约 60 分钟，首次会下载约 45MB 官方权重）
        Run configs/cnn/P0_pilot_R1.yaml
        Run configs/transfer/P0_pilot_ft.yaml
        Invoke-Py -m src.analysis.figures curves --name pilot_cnn configs/cnn/P0_pilot_R1.yaml:R1-lr0.05
        Invoke-Py -m src.analysis.figures curves --name pilot_ft configs/transfer/P0_pilot_ft.yaml:FT-lr0.003
        Write-Host "`n预实验完成。请根据 report/figures/pilot_*_curves.png 和 outputs/runs/P0_*/seed100/log.csv 确定轮数，例如：" -ForegroundColor Yellow
        Write-Host "  python -m src.select_decisions set epochs_32 100 --reason ""..."""
        Write-Host "  python -m src.select_decisions set epochs_224 15 --reason ""..."""
    }
    "2" {
        # 自定义 CNN 调参（3 次，约 1.2 小时；若 epochs_32=100，lr=0.05 那次直接复用预实验）
        Run configs/cnn/T_cnn.yaml
        Decide lr configs/cnn/T_cnn.yaml cnn_lr
    }
    "3" {
        # 逐项加组件（15 次，约 6 小时）：增强 → SE → 标签平滑，每一步按规则自动决策
        Run "configs/cnn/E1_*.yaml"
        Decide aug
        Run configs/cnn/E2_C3_se.yaml
        Decide attention
        Run configs/cnn/E3_C4_ls.yaml
        Decide loss
    }
    "4" {
        # 调度、早停、逐个去掉组件（最多约 13 次，约 5 小时；与已有配置相同的自动复用）
        Run configs/cnn/F_final.yaml "configs/cnn/E4_*.yaml" "configs/cnn/E5_*.yaml"
    }
    "5" {
        # 架构比较（约 8 小时）：两种模型各 3 点学习率调参，再各跑 3 个种子
        Run configs/arch/T_cnn_F.yaml configs/arch/T_resnet.yaml
        Decide-All (@(, @("lr", "configs/arch/T_cnn_F.yaml", "cnn_lr_F")) + @(, @("lr", "configs/arch/T_resnet.yaml", "resnet_lr")))
        Run "configs/arch/E6_*.yaml"
    }
    "6" {
        # 迁移学习（约 10 小时）：FT / S224 / LP / FT32 各 3 点调参，再各跑 3 个种子
        Run configs/transfer/T_ft.yaml configs/transfer/T_s224.yaml configs/transfer/T_lp.yaml configs/transfer/T_ft32.yaml
        Decide-All (@(, @("lr", "configs/transfer/T_ft.yaml", "ft_lr")) + @(, @("lr", "configs/transfer/T_s224.yaml", "s224_lr")) +
                    @(, @("lr", "configs/transfer/T_lp.yaml", "lp_lr")) + @(, @("lr", "configs/transfer/T_ft32.yaml", "ft32_lr")))
        Run "configs/transfer/E7*.yaml"
    }
    "7" {
        # 配置全部冻结后：统一测试集评估 + 全部 P0/P1 图表（约 30 分钟，主要是鲁棒性和 Grad-CAM 推理）
        Invoke-Py -m src.analysis.run_all
    }
    "8" {
        # P2 扩展（约 13 小时）：先做知识蒸馏（最重要，约 5 小时），再做 CBAM、CutMix、去 Dropout/BN、StepLR/OneCycle，最后分析。
        # 中途中断后重跑会跳过已完成的部分；只跑完蒸馏也足够写进报告。
        Run configs/ext/T_kd.yaml
        Decide kd configs/ext/T_kd.yaml
        Run configs/ext/E8_kd.yaml
        Run "configs/ext/X_*.yaml"
        Invoke-Py -m src.analysis.run_all --p2
    }
    default { throw "未知阶段：$Phase（可选 1–8 或 status）" }
}
Write-Host ("`n阶段 {0} 完成，用时 {1:N1} 小时。日志：{2}" -f $Phase, ((Get-Date) - $start).TotalHours, $Log) -ForegroundColor Green
