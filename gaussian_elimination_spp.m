% code from HW 1 with slight modification
function x = gaussian_elimination_spp(A, b)
    A = [A b];
    for i = 1:height(A)
        A = pivot_row(A,i);
        A = eliminate_column(A,i);
    end
    for i = height(A):-1:1
        A(i,:) = A(i,:) / A(i,i);
        for j = i - 1:-1:1
            A(j,:) = A(j,:) + A(i,:) * -1 * A(j,i);
        end
    end
    x = A(:,width(A));
end

function A = pivot_row(A,i)
    c = zeros(height(A), 1);
    for j = i:height(A)
        c(j) = abs(A(i,j)) ./ sum(abs(A(j,:)));
    end
    [~, max_row] = max(c);

    A([i max_row],:) = A([max_row i],:);
end

function A = eliminate_column(A,j)
    for i = (j+1):height(A)
        c = - A(i,j) ./ A(j,j);
        A(i,:) = A(i,:) + c * A(j,:);
    end
end