results = [];

% test random matrices
for n = 5:5:100
    A = make_random(n, 10);
    b = make_b(n, 10);
    desc = sprintf("Random %dx%d", n, n);
    results = test_matrix(A, b, desc, results);
end

% test well conditioned dense SDD matrices
for n = 5:5:100
    A = well_conditioned(n, 10, 10);
    b = make_b(n, 10);
    desc = sprintf("Well-Conditioned %dx%d", n, n);
    results = test_matrix(A, b, desc, results);
end

% test ill conditioned dense SDD matrices
for n = 5:5:100
    A = ill_conditioned(n, 100000, 100);
    b = make_b(n, 10);
    desc = sprintf("Ill-Conditioned %dx%d", n, n);
    results = test_matrix(A, b, desc, results);
end

% test sparse matrices
for n = 5:5:100
    A = make_sparse(n, 100, 0.1);
    b = make_b(n, 100);
    desc = sprintf("Sparse %dx%d", n, n);
    results = test_matrix(A, b, desc, results);
end

% test sparce SDD matrices
for n = 5:5:100
    A = make_SDD(make_sparse(n, 100, 0.1), 100);
    b = make_b(n, 100);
    desc = sprintf("Sparse SDD %dx%d", n, n);
    results = test_matrix(A, b, desc, results);
end

% test structured matrices
for n = 5:5:100
    A = make_structured(n, 20);
    b = make_b(n, 20);
    desc = sprintf("Structured %dx%d", n, n);
    results = test_matrix(A, b, desc, results);
end

writematrix(results, "results.csv");

function A = well_conditioned(n, s, kappa_max)
    condition = inf;
    while condition > kappa_max
        A = make_SDD(make_random(n, s), s);
        condition = cond(A);
    end
end

function A = ill_conditioned(n, s, kappa_min)
    condition = -inf;
    while condition < kappa_min
        A = make_SDD(make_random(n, s), s);
        condition = cond(A);
    end
end

function b = make_b(n, s)
    b = (rand(n, 1) - 0.5) .* 2 .* randi(s,n,1);
end

function A = make_SDD(A, s)
    for i = 1:size(A)
        A(i,i) = sum(abs(A(i,:)), 2) + randi(s);
    end
end

function A = make_structured(n, s)
    sum = 0;
    A = zeros(n, n);
    for i = 1:n
        random = randi(s .* 2) - s;
        sum = sum + abs(random);
        v = ones(n - i, 1) .* random;
        A = A + diag(v, i) + diag(v, -i);
    end
    random = randi(s);
    sum = sum + random;
    A = A + eye(n) .* sum;
end

function A = make_sparse(n, s, density)
    A = (full(sprand(n, n, density)) - 0.5) .* 2 .* randi(s, n);
end

function A = make_random(n, s)
    A = (rand(n) - 0.5) .* 2 .* randi(s, n, 1);
end

function results = test_matrix(A, b, desc, results)
    iters = 100;

    x_gauss_elim = gaussian_elimination_spp(A, b);
    x_gauss_seidel = gauss_seidel(A, b, ones(height(b),1), iters);

    absolute_error_gauss_elim = norm(A * x_gauss_elim - b, "inf");
    relative_error_gauss_elim = absolute_error_gauss_elim ./ norm(b, "inf");

    absolute_error_gauss_seidel = norm(A * x_gauss_seidel - b, "inf");
    relative_error_gauss_seidel = absolute_error_gauss_seidel ./ norm(b, "inf");

    time_gauss_elim = timeit(@() gaussian_elimination_spp(A, b));
    time_gauss_seidel = timeit(@() gauss_seidel(A, b, ones(height(b), 1), iters));
    
    % print to terminal
    fprintf("\n%s:\n", desc);
    fprintf("Gaussian Elimination:\n");
    fprintf("\tAbsolute Error: %f\n", absolute_error_gauss_elim);
    fprintf("\tRelative Error: %f\n", relative_error_gauss_elim);
    fprintf("\tTime: %fs\n", time_gauss_elim);
    fprintf("Gauss-Seidel Method:\n");
    fprintf("\tAbsolute Error: %f\n", absolute_error_gauss_seidel);
    fprintf("\tRelative Error: %f\n", relative_error_gauss_seidel);
    fprintf("\tTime: %fs\n", time_gauss_seidel);

    % return for later csv writing
    results = [results; absolute_error_gauss_elim, relative_error_gauss_elim, time_gauss_elim, absolute_error_gauss_seidel, relative_error_gauss_seidel, time_gauss_seidel];
end