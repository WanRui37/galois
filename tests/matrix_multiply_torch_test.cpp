#include "boost/scope/scope_exit.hpp"
#include "galois/op/matrix_multiply.hpp"
#include "galois/optimization/gemm_optimizer.hpp"
#include "tests/galois_test.hpp"
#include <torch/torch.h>

TEST(GaloisTests, TestMatrixMultiplyKernel) {
    auto native_cpu_info = optimization::NativeCpuInfo::Create();
    auto mat_mul_tile_policy = optimization::GemmTilePolicy::Create();
    auto [ir_mat_type_a, ir_mat_type_b, mat_mul_kernel] =
        mat_mul_tile_policy->Tile(ir::f32, native_cpu_info);
    ir_mat_type_a = ir_mat_type_a->value_type;
    ir_mat_type_b = ir_mat_type_b->value_type;

    fmt::print("a: {}, b: {}\n", ir_mat_type_a->name, ir_mat_type_b->name);

    auto ir_builder = ir::Builder::Create();
    ir_builder->matrix_multiply_kernel_queue.push_back(mat_mul_kernel);
    auto ir_packed_matrix_multiply_op_creator = op::MatrixMultiplyCreator::Create();
    auto ir_mat_type_c =
        ir_packed_matrix_multiply_op_creator->InferType({ir_mat_type_a, ir_mat_type_b});
    auto [ir_operator, scope_operator] = ir_builder->CreateOperator(
        ir::OperatorType::Create({ir_mat_type_a, ir_mat_type_b, ir_mat_type_c}, ir::void_),
        "matrix_multiply");
    ir_packed_matrix_multiply_op_creator->ExpressInline(
        ir_operator->inputs[0], ir_operator->inputs[1], ir_operator->inputs[2], ir_builder);
    scope_operator = nullptr;

    transform::Repeat(ir_operator, 1000000);

    auto ir_operator_counter = ir::OperationCounter::Create();
    auto operations = ir_operator_counter->CountOperation(ir_operator);

    auto jit_engine = jit::Engine::Create();
    auto mat_mul_fun =
        jit_engine->EmitOperatorSymbol<void (*)(void *, void *, void *)>(ir_operator);

    auto normalize_m = ir_mat_type_a->NormalizeShape()[0];
    auto normalize_k = ir_mat_type_a->NormalizeShape()[1];
    auto normalize_n = ir_mat_type_b->NormalizeShape()[1];

    auto sp_aligned256_mem_a = std::shared_ptr<void>(
        galois::auto_aligned_alloc(normalize_m * normalize_k * ir::f32->bytes),
        [](void *p) { free(p); });
    auto sp_aligned256_mem_b = std::shared_ptr<void>(
        galois::auto_aligned_alloc(normalize_k * normalize_n * ir::f32->bytes),
        [](void *p) { free(p); });

    auto sp_aligned256_mem_c = std::shared_ptr<void>(
        galois::auto_aligned_alloc(normalize_m * normalize_n * ir::f32->bytes),
        [](void *p) { free(p); });

    auto t0 = std::chrono::steady_clock::now();
    mat_mul_fun(sp_aligned256_mem_a.get(), sp_aligned256_mem_b.get(), sp_aligned256_mem_c.get());
    auto t1 = std::chrono::steady_clock::now();

    torch::Tensor tensor_a = torch::from_blob(
        sp_aligned256_mem_a.get(),
        {normalize_m, normalize_k},
        torch::TensorOptions().dtype(ir_f32)
    );

    torch::Tensor tensor_b = torch::from_blob(
        sp_aligned256_mem_b.get(),
        {normalize_k, normalize_n},
        torch::TensorOptions().dtype(ir_f32)
    );

    torch::Tensor tensor_c = torch::matmul(tensor_a, tensor_b);

    torch::Tensor tensor_c_custom = torch::from_blob(
        sp_aligned256_mem_c.get(),
        {normalize_m, normalize_n},
        torch::TensorOptions().dtype(ir_f32)
    );

    torch::Tensor abs_error = torch::abs(tensor_c - tensor_c_custom);
    float max_error = abs_error.max().item<float>();
    float mean_error = abs_error.mean().item<float>();

    fmt::print("LibTorch Time: {} μs\n", std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count());
    fmt::print("Max Error: {}\n", max_error);
    fmt::print("Mean Error: {}\n", mean_error);
    fmt::print("Results are close: {}\n", (max_error < 1e-5) ? "Yes" : "No");
}
