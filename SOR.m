format long

% build A
A = zeros(80, 80);
for i = 1:80
    for j = 1:80
        if j == i
            A(i,j) = 2 * i;
        elseif (j == i + 2 && i <= 78) || (j == i - 2 && i > 2)
            A(i,j) = 0.5 * i;
        elseif (j == i + 4 && i <= 76) || (j == i - 4 && i > 4)
            A(i,j) = 0.25 * i;
        end
    end
end

% build b
b = pi * ones(80,1);

x = SOR_method(A, b, 0.5, 10 ^ -5)

A * x

function x = SOR_method(A, b, omega, tolerance)
    n = size(A);
    x = zeros(80,1);
    condition = inf;

    while condition > tolerance
        for i = 1:n
            sigma = 0;
            for j = 1:n
                if j ~= i
                    sigma = sigma + A(i,j) * x(j);
                end
            end
            x(i) = (1 - omega) * x(i) + omega * (b(i) - sigma) / A(i,i);
        end
        condition = norm(A * x - b, "inf");
    end
end