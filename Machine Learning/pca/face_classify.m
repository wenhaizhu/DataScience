function face_classify(filepath)

img = load(filepath);
faceimage = img.image;
[firow, ficol] = size(faceimage);
vector = [];
for i=1:ficol
    vector = [vector; reshape(faceimage{i}',1, 2500)];
end

fprintf('for linear svm: \n');
ds=[20 50 100 200];
for i=1:4

    d = ds(i);
    fprintf('current d is %d :\n',d);
    eigenvecs = pca_fun(vector, d);
    data = [double(vector * eigenvecs) img.personID' img.subsetID'];
    meanaccuford = 0;
    for setid=1:5
        fprintf('%d is test set\n',setid);
        id = [1 2 3 4 5];
        id = id(id ~= setid);
        train_set = data(find(data(:,end)~=setid),:);
        train_label = train_set(:,end-1);
        train_data = train_set(:,1:end-2);
        test_set = data(find(data(:,end)==setid),:);
        test_label = test_set(:,end-1);
        test_data = test_set(:,1:end-2);
        C = [4^-6 4^-5 4^-4 4^-3 4^-2 4^-1 1 4 4^2 ];
        c_accu = [];
        for ci=1:size(C,2)
            cur_accu = 0;
            for trainset_id=1:4
                curtrain_id = id(trainset_id);
                subtrain_set = train_set(find(train_set(:,end)~=curtrain_id),:);
                subtrain_label = subtrain_set(:,end-1);
                subtrain_data = subtrain_set(:,1:end-2);
                
                subvalid_set = train_set(find(train_set(:,end)==curtrain_id),:);
                subvalid_label = subvalid_set(:,end-1);
                subvalid_data = subvalid_set(:,1:end-2);
                
                opts = sprintf('-q -t 0 -c %f',C(1,ci));
                model = svmtrain(subtrain_label, subtrain_data, opts);
                [predicted_label, accu, decision] = svmpredict(subvalid_label, subvalid_data, model, '-q');
                cur_accu = cur_accu + accu(1);
            end
            c_accu = [c_accu cur_accu/4];
        end
        [maxaccu, index] = max(c_accu);
        bestc = C(1,index);
        fprintf('best c is %f, its accuracy is %f \n', bestc, maxaccu);
        opts = sprintf('-q -t 0 -c %f',bestc);
        bestmodel = svmtrain(train_label, train_data, opts);
        [predicted_label, accu, decision] = svmpredict(test_label, test_data, bestmodel, '-q');
        meanaccuford = meanaccuford + accu(1);
    end
    meanaccu = meanaccuford / 5;

    fprintf('the average accuracy is %f \n', meanaccu);


end


fprintf('\n\n\n-------------------\nfor RBF kernel svm: \n');
ds=[20 50 100 200];
for i=1:4

    d = ds(i);
    fprintf('current d is %d :\n',d);
    eigenvecs = pca_fun(vector, d);
    data = [double(vector * eigenvecs) img.personID' img.subsetID'];
    meanaccuford = 0;
    for setid=1:5
        fprintf('%d is test set\n',setid);
        id = [1 2 3 4 5];
        id = id(id ~= setid);
        train_set = data(find(data(:,end)~=setid),:);
        train_label = train_set(:,end-1);
        train_data = train_set(:,1:end-2);
        test_set = data(find(data(:,end)==setid),:);
        test_label = test_set(:,end-1);
        test_data = test_set(:,1:end-2);
        C = [4^-6 4^-5 4^-4 4^-3 4^-2 4^-1 1 4 4^2 ];
        gamma = [4^-7 4^-6 4^-5 4^-4 4^-3 4^-2 4^-1];
        whole_accu = [];
        for ci=1:size(C,2)
            c_accu = [];
            for gi=1:size(gamma,2)
                cur_accu = 0;
                for trainset_id=1:4
                    curtrain_id = id(trainset_id);
                    subtrain_set = train_set(find(train_set(:,end)~=curtrain_id),:);
                    subtrain_label = subtrain_set(:,end-1);
                    subtrain_data = subtrain_set(:,1:end-2);

                    subvalid_set = train_set(find(train_set(:,end)==curtrain_id),:);
                    subvalid_label = subvalid_set(:,end-1);
                    subvalid_data = subvalid_set(:,1:end-2);

                    opts = sprintf('-q -t 2 -c %f -g %f',C(1,ci), gamma(1,gi));
                    model = svmtrain(subtrain_label, subtrain_data, opts);
                    [predicted_label, accu, decision] = svmpredict(subvalid_label, subvalid_data, model, '-q');
                    cur_accu = cur_accu + accu(1);
                end
                c_accu = [c_accu cur_accu/4];
            end
            whole_accu = [whole_accu; c_accu];
        end
        [v,ind]=max(whole_accu);
        [maxaccu,ind1]=max(max(whole_accu)); 
        bestcc = C(1,ind(ind1));
        bestgamma = gamma(1, ind1);
        fprintf('best c is %f, bestgamma is %f, its accuracy is %f \n', bestcc, bestgamma, maxaccu);
        opts = sprintf('-q -t 2 -c %f -g %f',bestcc, bestgamma);
        bestmodel = svmtrain(train_label, train_data, opts);
        [predicted_label, accu, decision] = svmpredict(test_label, test_data, bestmodel, '-q');
        meanaccuford = meanaccuford + accu(1);
    end
    meanaccu = meanaccuford / 5;

    fprintf('the average accuracy is %f \n', meanaccu);


end
