data1 = load('ex1data2.txt');
X1 = data1(:, 1:2);
y1 = data1(:, 3);
mu1= mean(X1,1);
sigma1 = std(X1,0,1);
X_norm = X1;
for i= 1:1:size(X1,1)
    X_norm(i,1)=(X_norm(i,1)-mu1(1))./sigma1(1);
    X_norm(i,2)=(X_norm(i,2)-mu1(2))./sigma1(2);

end