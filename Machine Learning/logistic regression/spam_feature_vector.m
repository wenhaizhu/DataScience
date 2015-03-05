function features = spam_feature_vector(filepath, dic)

features = [];
[dic_row, dic_col] = size(dic);
A = dir(fullfile(filepath,'*.txt'));
num = size(A);
for i = 1:num
    filename = A(i).name;
    text = fileread(strcat(filepath, filename));
    %disp(text);
    lines = strread(text,'%s','delimiter','\n');
    [lines_row, lines_col] = size(lines);
    feature = zeros(1, dic_col);
    for k = 1:lines_row
        line = lines(k);
        %disp(line{1});
        [tokens, matches] = strsplit(line{1},{' ','.',',','?'},'CollapseDelimiters',true);
        [tokens_row, tokens_col] = size(tokens);
        for j = 1:tokens_col
            %disp(lower(tokens(j)));
            index = find(ismember(dic,lower(tokens(j))));
            if (isempty(index) ~= 1)
                feature(index) = feature(index) + 1;
            end
        end
    end
    features = [features;feature];
end