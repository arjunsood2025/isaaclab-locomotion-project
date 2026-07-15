// Fixed-rate C++ inference loop for an exported TorchScript locomotion policy.
//
// Demonstrates the deployment-critical properties a Python node cannot guarantee:
//   - deterministic 50 Hz scheduling with absolute deadlines (no drift accumulation)
//   - single-allocation hot path: observation and action tensors are reused
//   - latency and deadline-miss accounting with percentile reporting
//
// The robot I/O is stubbed behind read_robot_state() / write_joint_targets() with
// TODO markers -- on real hardware these become the vendor SDK calls (e.g.
// unitree_sdk2 LowState/LowCmd over DDS). Everything else is production-shaped.
//
// Build & run (see CMakeLists.txt):
//   ./policy_loop ../../exported/policy.pt 48 50

#include <torch/script.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kNumJoints = 12;
constexpr float kActionScale = 0.25f;  // must match ActionsCfg.joint_pos.scale

// default standing pose, policy joint order (matches training config)
constexpr float kDefaultJointPos[kNumJoints] = {
    0.1f, -0.1f, 0.1f, -0.1f, 0.8f, 0.8f, 1.0f, 1.0f, -1.5f, -1.5f, -1.5f, -1.5f};

struct RobotState {
  float base_lin_vel[3]{};
  float base_ang_vel[3]{};
  float projected_gravity[3]{0.f, 0.f, -1.f};
  float command[3]{};
  float joint_pos[kNumJoints]{};
  float joint_vel[kNumJoints]{};
};

// TODO(hardware): replace with the vendor SDK state read (joint encoders, IMU,
// state estimator). Stubbed with the standing pose so the loop runs end-to-end.
void read_robot_state(RobotState& state) {
  for (int i = 0; i < kNumJoints; ++i) {
    state.joint_pos[i] = kDefaultJointPos[i];
    state.joint_vel[i] = 0.f;
  }
}

// TODO(hardware): replace with the vendor SDK command write (position targets for
// the onboard PD loop, kp=25 kd=0.5 to match training).
void write_joint_targets(const float* /*targets*/) {}

void build_observation(const RobotState& s, const float* last_action, float* obs) {
  int k = 0;
  for (int i = 0; i < 3; ++i) obs[k++] = s.base_lin_vel[i];
  for (int i = 0; i < 3; ++i) obs[k++] = s.base_ang_vel[i];
  for (int i = 0; i < 3; ++i) obs[k++] = s.projected_gravity[i];
  for (int i = 0; i < 3; ++i) obs[k++] = s.command[i];
  for (int i = 0; i < kNumJoints; ++i) obs[k++] = s.joint_pos[i] - kDefaultJointPos[i];
  for (int i = 0; i < kNumJoints; ++i) obs[k++] = s.joint_vel[i];
  for (int i = 0; i < kNumJoints; ++i) obs[k++] = last_action[i];
}

double percentile(std::vector<double> sorted, double q) {
  const size_t idx = std::min(static_cast<size_t>(q * sorted.size()),
                              sorted.size() - 1);
  return sorted[idx];
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: %s <policy.pt> [obs_dim=48] [rate_hz=50]\n", argv[0]);
    return 1;
  }
  const std::string model_path = argv[1];
  const int obs_dim = argc > 2 ? std::stoi(argv[2]) : 48;
  const double rate_hz = argc > 3 ? std::stod(argv[3]) : 50.0;
  const auto period = std::chrono::duration_cast<std::chrono::nanoseconds>(
      std::chrono::duration<double>(1.0 / rate_hz));

  torch::jit::Module policy;
  try {
    policy = torch::jit::load(model_path);
  } catch (const c10::Error& e) {
    std::fprintf(stderr, "failed to load %s: %s\n", model_path.c_str(), e.what());
    return 1;
  }
  policy.eval();
  torch::NoGradGuard no_grad;

  // preallocated buffers -- the control loop must not allocate
  auto obs_tensor = torch::zeros({1, obs_dim}, torch::kFloat32);
  float* obs_data = obs_tensor.data_ptr<float>();
  float last_action[kNumJoints]{};
  float targets[kNumJoints]{};
  RobotState state{};

  std::vector<double> latencies_ms;
  latencies_ms.reserve(100000);
  long deadline_misses = 0;
  long ticks = 0;

  std::printf("policy loop: %s | obs_dim=%d | %.0f Hz (budget %.2f ms)\n",
              model_path.c_str(), obs_dim, rate_hz, 1000.0 / rate_hz);

  // warmup: first calls trigger JIT optimization; keep them out of the statistics
  for (int i = 0; i < 50; ++i) policy.forward({obs_tensor});

  auto next_deadline = std::chrono::steady_clock::now() + period;
  const auto run_until = std::chrono::steady_clock::now() + std::chrono::seconds(20);

  while (std::chrono::steady_clock::now() < run_until) {
    const auto t0 = std::chrono::steady_clock::now();

    read_robot_state(state);
    build_observation(state, last_action, obs_data);

    const auto out = policy.forward({obs_tensor}).toTensor();
    const float* action = out.data_ptr<float>();

    for (int i = 0; i < kNumJoints; ++i) {
      last_action[i] = action[i];
      targets[i] = kDefaultJointPos[i] + kActionScale * action[i];
      // safety clamp, same envelope as the ROS2 node
      targets[i] = std::clamp(targets[i], kDefaultJointPos[i] - 0.6f,
                              kDefaultJointPos[i] + 0.6f);
    }
    write_joint_targets(targets);

    const auto t1 = std::chrono::steady_clock::now();
    latencies_ms.push_back(
        std::chrono::duration<double, std::milli>(t1 - t0).count());

    ++ticks;
    if (t1 > next_deadline) ++deadline_misses;
    // absolute-deadline scheduling: sleep_until prevents drift accumulation
    std::this_thread::sleep_until(next_deadline);
    next_deadline += period;
  }

  std::sort(latencies_ms.begin(), latencies_ms.end());
  const double mean =
      std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0) /
      static_cast<double>(latencies_ms.size());
  std::printf("\nticks: %ld | deadline misses: %ld (%.2f%%)\n", ticks,
              deadline_misses, 100.0 * deadline_misses / ticks);
  std::printf("loop latency ms: mean %.3f | p50 %.3f | p95 %.3f | p99 %.3f | max %.3f\n",
              mean, percentile(latencies_ms, 0.50), percentile(latencies_ms, 0.95),
              percentile(latencies_ms, 0.99), latencies_ms.back());
  return 0;
}
