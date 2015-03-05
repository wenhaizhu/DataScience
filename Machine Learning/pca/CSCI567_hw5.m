function CSCI567_hw5()

fprintf('for 4(c): 5 eigenfaces:\n');

filepath='./face_data.mat';
get5faces(filepath);
fprintf('images will show after the program finishes running\n');
fprintf('for 4(d):\n');
face_classify(filepath);
