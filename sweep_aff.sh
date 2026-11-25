#!/bin/bash
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8
PY=~/miniconda3/envs/dl_lab/bin/python
r() { # $1=label  $2..=env 赋值  然后 nt
  local lab="$1"; shift; local nt="$1"; shift
  echo "--- $lab  (nt=$nt) ---"
  env -u OMP_PROC_BIND -u OMP_PLACES -u GOMP_CPU_AFFINITY -u OMP_WAIT_POLICY "$@" \
      ~/miniconda3/envs/dl_lab/bin/python bench_affinity.py $nt "$lab" 2>&1 \
      | grep -v "^USDT\|正在编译"
}
r "0 基线(全不绑)"        6
r "1 PLACES=cores SPREAD" 6 OMP_PLACES=cores OMP_PROC_BIND=spread
r "2 PLACES=cores CLOSE"  6 OMP_PLACES=cores OMP_PROC_BIND=close
r "3 GOMP_AFF=0-5(不开BIND)" 6 GOMP_CPU_AFFINITY=0-5
r "4 GOMP_AFF=1-5(留核0)"    6 GOMP_CPU_AFFINITY=1-5
r "5 PLACES=cores SPREAD nt5" 5 OMP_PLACES=cores OMP_PROC_BIND=spread
r "6 GOMP_AFF=1-5 nt5"       5 GOMP_CPU_AFFINITY=1-5
