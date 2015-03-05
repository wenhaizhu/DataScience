function CSCI567_hw4()
data = csvread('./2DGaussian.csv',1);
input = data(:,2:3);

k=[2,3,5];
for i = 1:size(k,2)
    curk = k(1,i);
    [class, means] = kmeans(input, curk);
    totaldata = [input class];
    figure();
    for j = 1:curk
        hold on;
        scatter(totaldata(totaldata(:,end)==j,1:1),totaldata(totaldata(:,end)==j,2:2),'d');
    end
end

iteration = 50;
curk = 4;
iter_array = [1:50];
figure();
for i=1:5
    [class, means, funcvalue] = kmeans2(input, curk, iteration);
    hold on;
    if i == 1
        plot(iter_array, funcvalue, '--');
    elseif i == 2
        plot(iter_array, funcvalue, '-.');
    elseif i == 3
        plot(iter_array, funcvalue, '-x');    
    elseif i == 4
        plot(iter_array, funcvalue, '-o');    
    elseif i == 5
        plot(iter_array, funcvalue, '-+');    
    end
    title('plot');
    xlabel('Iteration');
    ylabel('Objective function value');
end

k=[3,8,15];
pic = './hw4.jpg';
for i = 1:size(k,2)
    curk = k(1,i);
    vecter_quan(pic, curk);
end