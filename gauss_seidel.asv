
function x = gauss_seidel(A, b, x, iters)
    for k = 1:iters
        for i = 1:size(A,1)
            x(i) = (b(i) - sum(A(i,:)' .* x) + A(i,i) * x(i)) / A(i,i);
        end
    end
end