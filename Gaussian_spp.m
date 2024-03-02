
A = [3.3330,15920,10.333,7953;2.2220,16.710,9.6120,0.965;-1.5611,5.1792,-1.6855,2.714]

A = spp(A)
A = reduce(A)
A(:,4)

function A = pivot_row(A,i)
    max_val = -inf;
    max_row = -inf;
    for j = i:height(A)
        val = abs(A(i,j)) ./ sum(A(j,:));
        if val > max_val
            max_val = val;
            max_row = j;
        end
    end
    A([i max_row],:) = A([max_row i],:);
end

function A = eliminate_column(A,j)
    for i = (j+1):height(A)
        c = - A(i,j) ./ A(j,j);
        A(i,:) = A(i,:) + c * A(j,:);
    end
end

function A = spp(A)
    for i = 1:height(A)
        A = pivot_row(A,i);
        A = eliminate_column(A,i);
    end
end

function A = reduce(A)
    for j = height(A):-1:1
        for i = (j-1):-1:1
            c = - A(i,j) ./ A(j,j);
            A(i,:) = A(i,:) + c * A(j,:);
        end
        c = 1 ./ A(j,j);
        A(j,:) = c * A(j,:);
    end
end