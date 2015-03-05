function [data, label] = clean_data(filename)
    fid = fopen(filename);
    out = textscan(fid,'%s%s%s%s%s%s%s','delimiter',',');
    fclose(fid);

    train_data = [];
    [row, col] = size(out);
    [col_row, col_col] = size(out{1});

    for i=1:2
        temp = [];
        for j=1:col_row
            if isequal(out{i}{j},'vhigh')
                temp = [temp; [1,0,0,0]];
            elseif isequal(out{i}{j},'high')
                temp = [temp; [0,1,0,0]];
            elseif isequal(out{i}{j},'med')
                temp = [temp; [0,0,1,0]];
            elseif isequal(out{i}{j},'low')
                temp = [temp; [0,0,0,1]];
            end
        end    
        train_data = [train_data, temp];
    end
    temp = [];
    for j=1:col_row

        if isequal(out{3}{j},'2')
            temp = [temp; [1,0,0,0]];
        elseif isequal(out{3}{j},'3')
            temp = [temp; [0,1,0,0]];
        elseif isequal(out{3}{j},'4')
            temp = [temp; [0,0,1,0]];
        elseif isequal(out{3}{j},'5more')
            temp = [temp; [0,0,0,1]];
        end
    end    
    train_data = [train_data, temp];    
    temp = [];    
    for j=1:col_row
        if isequal(out{4}{j},'2')
            temp = [temp; [1,0,0]];
        elseif isequal(out{4}{j},'4')
            temp = [temp; [0,1,0]];
        elseif isequal(out{4}{j},'more')
            temp = [temp; [0,0,1]];
        end
    end    
    train_data = [train_data, temp];    
    temp = [];
    for j=1:col_row
        if isequal(out{5}{j},'small')
            temp = [temp; [1,0,0]];
        elseif isequal(out{5}{j},'med')
            temp = [temp; [0,1,0]];
        elseif isequal(out{5}{j},'big')
            temp = [temp; [0,0,1]];
        end
    end    
    train_data = [train_data, temp];  
    temp = [];
    for j=1:col_row
        if isequal(out{6}{j},'low')
            temp = [temp; [1,0,0]];
        elseif isequal(out{6}{j},'med')
            temp = [temp; [0,1,0]];
        elseif isequal(out{6}{j},'high')
            temp = [temp; [0,0,1]];
        end
    end    
    train_data = [train_data, temp];

    train_label = [];
    for j=1:col_row
        if isequal(out{7}{j},'unacc')
            train_label = [train_label; [1]];
        elseif isequal(out{7}{j},'acc')
            train_label = [train_label; [2]];
        elseif isequal(out{7}{j},'good')
            train_label = [train_label; [3]];
        elseif isequal(out{7}{j},'vgood')
            train_label = [train_label; [4]];
        end
    end    
    data = train_data;
    label = train_label;

    