#include <DemBones/DemBones.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>

int main() {
    using Solver = Dem::DemBones<double, double>;
    using Clock = std::chrono::steady_clock;

    Solver solver;
    solver.nV = 4;
    solver.nB = 2;
    solver.nS = 1;
    solver.nF = 2;
    solver.nWeightsIters = 20;
    solver.nnz = 2;
    solver.weightsSmooth = 0.0;
    solver.fStart.resize(2);
    solver.fStart << 0, solver.nF;
    solver.subjectID = Eigen::VectorXi::Zero(solver.nF);
    solver.lockM = Eigen::VectorXi::Ones(solver.nB);
    solver.lockW = Solver::VectorX::Zero(solver.nV);

    solver.u.resize(3, solver.nV);
    solver.u << 0.0, 1.0, 2.0, 3.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0;
    solver.fv = {{0, 1, 2}, {1, 2, 3}};

    solver.m = Solver::Matrix4::Identity().replicate(solver.nF, solver.nB);
    solver.m.block<4, 4>(4, 4)(0, 3) = 1.0; // frame 1, bone 1: +1 metre X

    const Eigen::RowVector4d truth(0.0, 0.25, 0.75, 1.0);
    solver.v.resize(3 * solver.nF, solver.nV);
    solver.v.topRows(3) = solver.u;
    solver.v.bottomRows(3) = solver.u;
    solver.v.row(3) += truth;

    const auto start = Clock::now();
    solver.computeWeights();
    const auto elapsed = std::chrono::duration<double, std::milli>(Clock::now() - start).count();

    Eigen::MatrixXd dense(solver.w);
    const double maxError = (dense.row(1) - truth).cwiseAbs().maxCoeff();
    std::cout << "dem-bones: solve_ms=" << elapsed << " max_weight_error=" << maxError << "\n";
    std::cout << dense << "\n";
    return std::isfinite(maxError) && maxError < 1e-4 ? 0 : 1;
}

