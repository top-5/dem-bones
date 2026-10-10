#include <DemBones/DemBones.h>
#include <cmath>
#include <exception>
#include <stdexcept>
#include <string>

namespace { thread_local std::string last_error; }

extern "C" {
const char* dem_last_error() { return last_error.c_str(); }

// All arrays are row-major. Output is vertex-major, then bone.
int dem_solve_weights(const double* rest, const unsigned int* faces, const double* targets,
                      const double* transforms, const double* initial_weights, const double* lock_weights,
                      double* output, int vertex_count, int face_count,
                      int frame_count, int bone_count,
                      int max_influences, int iterations, double smoothness) {
  try {
    if (!rest || !faces || !targets || !transforms || !output) throw std::invalid_argument("null array pointer");
    if (vertex_count <= 0 || face_count <= 0 || frame_count <= 0 || bone_count <= 0) throw std::invalid_argument("counts must be positive");
    if (max_influences <= 0 || max_influences > bone_count) throw std::invalid_argument("invalid max influences");
    if (iterations <= 0 || smoothness < 0) throw std::invalid_argument("invalid solver configuration");
    using Solver = Dem::DemBones<double, double>;
    Solver solver;
    solver.nV = vertex_count; solver.nF = frame_count; solver.nB = bone_count; solver.nS = 1;
    solver.nnz = max_influences; solver.nWeightsIters = iterations; solver.weightsSmooth = smoothness;
    solver.fStart.resize(2); solver.fStart << 0, frame_count;
    solver.subjectID = Eigen::VectorXi::Zero(frame_count);
    solver.lockM = Eigen::VectorXi::Ones(bone_count);
    solver.lockW = Solver::VectorX::Zero(vertex_count);
    if (lock_weights && !initial_weights) throw std::invalid_argument("lock weights require initial weights");
    if (initial_weights) {
      std::vector<Solver::Triplet, Eigen::aligned_allocator<Solver::Triplet>> weights;
      weights.reserve(vertex_count * max_influences);
      for (int vertex = 0; vertex < vertex_count; ++vertex) {
        double sum = 0;
        for (int bone = 0; bone < bone_count; ++bone) {
          const double value = initial_weights[vertex * bone_count + bone];
          if (!std::isfinite(value) || value < 0) throw std::invalid_argument("invalid initial weight");
          sum += value;
          if (value > 0) weights.emplace_back(bone, vertex, value);
        }
        if (sum <= 0) throw std::invalid_argument("initial weights contain an empty vertex");
        if (lock_weights) {
          const double value = lock_weights[vertex];
          if (!std::isfinite(value) || value < 0 || value > 1) throw std::invalid_argument("invalid weight lock");
          solver.lockW(vertex) = value;
        }
      }
      solver.w.resize(bone_count, vertex_count);
      solver.w.setFromTriplets(weights.begin(), weights.end());
    }
    solver.u.resize(3, vertex_count);
    for (int vertex = 0; vertex < vertex_count; ++vertex)
      for (int axis = 0; axis < 3; ++axis) solver.u(axis, vertex) = rest[vertex * 3 + axis];
    solver.fv.resize(face_count);
    for (int face = 0; face < face_count; ++face) {
      solver.fv[face] = {static_cast<int>(faces[face * 3]), static_cast<int>(faces[face * 3 + 1]),
                         static_cast<int>(faces[face * 3 + 2])};
      for (int corner : solver.fv[face])
        if (corner < 0 || corner >= vertex_count) throw std::invalid_argument("face index outside vertex table");
    }
    solver.v.resize(frame_count * 3, vertex_count);
    for (int frame = 0; frame < frame_count; ++frame)
      for (int vertex = 0; vertex < vertex_count; ++vertex)
        for (int axis = 0; axis < 3; ++axis)
          solver.v(frame * 3 + axis, vertex) = targets[(frame * vertex_count + vertex) * 3 + axis];
    solver.m.resize(frame_count * 4, bone_count * 4);
    for (int frame = 0; frame < frame_count; ++frame)
      for (int bone = 0; bone < bone_count; ++bone)
        for (int row = 0; row < 4; ++row)
          for (int column = 0; column < 4; ++column)
            solver.m(frame * 4 + row, bone * 4 + column) = transforms[((frame * bone_count + bone) * 4 + row) * 4 + column];
    solver.computeWeights();
    Eigen::MatrixXd dense(solver.w);
    for (int vertex = 0; vertex < vertex_count; ++vertex)
      for (int bone = 0; bone < bone_count; ++bone) output[vertex * bone_count + bone] = dense(bone, vertex);
    last_error.clear();
    return 0;
  } catch (const std::exception& error) { last_error = error.what(); return 1; }
  catch (...) { last_error = "unknown native exception"; return 2; }
}
}
