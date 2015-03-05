function generate_data(train_data, train_label)
fid = fopen('data.txt','wt');
[data_row data_col] = size(train_data);
for i = 1:data_row
    fprintf(fid,'%s ',num2str(train_label(i,1)));
    for j = 1:data_col
        str = strcat(num2str(j), ':', num2str(train_data(i,j)));
        
        fprintf(fid,'%s ',str);
    end
    
    fprintf(fid,'\n');
end
fclose(fid)
