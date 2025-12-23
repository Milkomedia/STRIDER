# STRIDER MPC (MuJoCo + ACADOS NMPC + Geometric Control)

가변형상(모핑) 쿼드로터 **STRIDER/JellyFish** 모델을 MuJoCo에서 시뮬레이션하고,
- C++ 실시간 **Geometric(SE(3)) Controller**(+ DoB/CoM 추정/할당)
- Python(acados/casadi) 기반 **NMPC(arm shaping / joint command)**

를 **멀티스레드**로 결합해 실행하는 연구/개발용 프로젝트입니다.

> 실행 중 **SPACE** 키로 MPC ON/OFF 토글 가능 (MuJoCo Viewer 창 포커스 필요)

---

## 주요 기능

- **MuJoCo 모델 기반 실시간 시뮬레이션**
  - `resources/mujoco/scene.xml` → `JellyFish.xml` 포함
  - STL 메쉬 포함 (`resources/mujoco/*.STL`)
- **실시간 제어 루프**
  - 시뮬레이션 스텝: `SIM_HZ=1000 Hz`
  - 제어 주기: `CTRL_HZ=400 Hz`
  - 뷰어 렌더: `VIEWER_HZ=30 Hz`
- **Geometric Controller + 제어 할당**
  - `modules/geometry_ctrl/` (FDCL 계열 SE(3) 제어기)
  - `FC::ControllerGeom`에서 쿼드로터용 출력(PWM/틸트/렌치) 생성
- **IK/FK**
  - 4개 팔(각 5-DOF)의 DH 기반 FK/IK (`include/utils.hpp`)
  - FK로 현재 `{b}->{cot}` 평균 위치 추정, IK로 팔 조인트 목표각 계산
- **MPC(옵션)**
  - C++ ↔ Python(pybind11 embed)로 NMPC 호출 (`src/mpc_wrapper.cpp`)
  - `resources/mpc_py/`에서 acados OCP 구성/풀이

---

## 레포 구조 (핵심만)

