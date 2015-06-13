%% STEP 0: Initialize Parameters and Load Data

imageDim = 28;
numClasses = 10;

filterDim1 = 5;
numFilters1 = 10;
poolDim1 = 2;

filterDim2 = 5;
numFilters2 = 10;
poolDim2 = 2;

% Load MNIST Train
%training data
addpath ./common/;
images = loadMNISTImages('./common/train-images-idx3-ubyte');
images = reshape(images,imageDim,imageDim,[]);
labels = loadMNISTLabels('./common/train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

%test data
testImages = loadMNISTImages('./common/t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('./common/t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

%Initialize Parameters
Wc1 = 1e-1*randn(filterDim1,filterDim1,numFilters1);
bc1 = zeros(numFilters1, 1);
outDim1 = imageDim - filterDim1 + 1;
outDim1 = outDim1 / poolDim1;

Wc2 = 1e-1*randn(filterDim2,filterDim2,numFilters1,numFilters2);
bc2 = zeros(numFilters2,1);
outDim2 = outDim1 - filterDim2 + 1;
outDim2 = outDim2 / poolDim2;

hiddenSize = outDim2 ^ 2 * numFilters2;

r  = sqrt(6) / sqrt(numClasses+hiddenSize+1);
Wd = rand(numClasses, hiddenSize) * 2 * r - r;
bd = zeros(numClasses, 1);

%% STEP 1: Learn Parameters
epochs = 8;
alpha = 0.08;
minibatch = 50;

% Setup for momentum
mom = 0.5;

Wc1_velocity = zeros(size(Wc1));
bc1_velocity = zeros(size(bc1));
Wc2_velocity = zeros(size(Wc2));
bc2_velocity = zeros(size(bc2));
Wd_velocity = zeros(size(Wd));
bd_velocity = zeros(size(bd));

lambda = 0.0001;
m = length(labels);

%%SGD loop
it = 0;
for e = 1:epochs
    rp = randperm(m);
    
    for s=1:minibatch:(m-minibatch+1)
        it = it + 1;

        % get next randomly selected minibatch
        mb_data = images(:,:,rp(s:s+minibatch-1));
        mb_labels = labels(rp(s:s+minibatch-1));
        numImages = length(mb_labels);
        
        convDim1 = imageDim - filterDim1 + 1;
        outputDim1 = convDim1 / poolDim1;
        convDim2 = outputDim1 - filterDim2 + 1;
        outputDim2 = convDim2 / poolDim2;
        
        %FeedForward Propagation
        %first convolve and pooling
        activations1 = cnnConvolve(filterDim1, numFilters1, mb_data, Wc1, bc1);%sigmoid(wx+b)
        activationsPooled1 = cnnPool(poolDim1, activations1);
        
        %second convolve and pooling
        activations2 = cnnConvolve4D(activationsPooled1, Wc2, bc2);%sigmoid(wx+b)
        activationsPooled2 = cnnPool(poolDim2, activations2);
        
        activationsPooled2 = reshape(activationsPooled2,[],numImages);
        h = exp(bsxfun(@plus,Wd * activationsPooled2,bd));
        probs = bsxfun(@rdivide,h,sum(h,1));  
        
        %Caculate Cost
        logp = log(probs);
        index = sub2ind(size(logp),mb_labels',1:size(probs,2));
        ceCost = -sum(logp(index));
        wCost = lambda/2 * (sum(Wd(:).^2)+sum(Wc1(:).^2)+sum(Wc2(:).^2));
        cost = ceCost/numImages + wCost;
        
        %Backpropagation
        %Deltas
        output = zeros(size(probs));
        output(index) = 1;
        DeltaSoftmax = probs - output;
        
        DeltaPool2 = reshape(Wd' * DeltaSoftmax,outputDim2,outputDim2,numFilters2,numImages);
        DeltaUnpool2 = zeros(convDim2,convDim2,numFilters2,numImages);
        
        for imNum = 1:numImages
            for FilterNum = 1:numFilters2
                unpool = DeltaPool2(:,:,FilterNum,imNum);
                DeltaUnpool2(:,:,FilterNum,imNum) = kron(unpool,ones(poolDim2))./(poolDim2 ^ 2);
            end
        end
        
        DeltaConv2 = DeltaUnpool2 .* activations2 .* (1 - activations2);
        DeltaPool1 = zeros(outputDim1,outputDim1,numFilters1,numImages);
        for i = 1:numImages
            for f1 = 1:numFilters1
                for f2 = 1:numFilters2
                    DeltaPool1(:,:,f1,i) = DeltaPool1(:,:,f1,i) + convn(DeltaConv2(:,:,f2,i),Wc2(:,:,f1,f2),'full');
                end
            end
        end
        
        DeltaUnpool1 = zeros(convDim1,convDim1,numFilters1,numImages);
        for imNum = 1:numImages
            for FilterNum = 1:numFilters1
                unpool = DeltaPool1(:,:,FilterNum,imNum);
                DeltaUnpool1(:,:,FilterNum,imNum) = kron(unpool,ones(poolDim1))./(poolDim1 ^ 2);
            end
        end
        
        DeltaConv1 = DeltaUnpool1 .* activations1 .* (1-activations1);
        
        %Gradients
        Wd_grad = (1./numImages) .* DeltaSoftmax*activationsPooled2'+lambda*Wd;
        bd_grad = (1./numImages) .* sum(DeltaSoftmax,2);
        
        %second convolution layer
        Wc2_grad = zeros(size(Wc2));
        bc2_grad = zeros(size(bc2));
        for fil2 = 1:numFilters2
            for fil1 = 1:numFilters1
                for im = 1:numImages
                    Wc2_grad(:,:,fil1,fil2) = Wc2_grad(:,:,fil1,fil2) + conv2(activationsPooled1(:,:,fil1,im),rot90(DeltaConv2(:,:,fil2,im),2),'valid');
                end
                Wc2_grad(:,:,fil1,fil2) = Wc2_grad(:,:,fil1,fil2) * (1/numImages);
            end
            temp = DeltaConv2(:,:,fil2,:);
            bc2_grad(fil2) = (1/numImages)*sum(temp(:));
        end
        
        %first convolutional layer
        Wc1_grad = zeros(size(Wc1));
        bc1_grad = zeros(size(bc1));
        for fil1 = numFilters1
            for im = 1:numImages
                Wc1_grad(:,:,fil1) = Wc1_grad(:,:,fil1) + conv2(images(:,:,im),rot90(DeltaConv1(:,:,fil1,im),2),'valid');
            end
            Wc1_grad(:,:,fil1) = Wc1_grad(:,:,fil1) * (1/numImages);
            temp = DeltaConv1(:,:,fil1,:);
            bc1_grad(fil1) = (1/numImages) * sum(temp(:));
        end
       
        
        Wd_velocity = mom*Wd_velocity - alpha*Wd_grad - lambda*Wd;
        bd_velocity = mom*bd_velocity - alpha*bd_grad - lambda*bd;
        Wc2_velocity = mom*Wc2_velocity - alpha*Wc2_grad - lambda*Wc2;
        bc2_velocity = mom*bc2_velocity - alpha*bc2_grad - lambda*bc2;
        Wc1_velocity = mom*Wc1_velocity - alpha*Wc1_grad - lambda*Wc1;
        bc1_velocity = mom*bc1_velocity - alpha*bc1_grad - lambda*bc1;
                        
        Wd = Wd + Wd_velocity;
        bd = bd + bd_velocity;
        Wc2 = Wc2 + Wc2_velocity;
        bc2 = bc2 + bc2_velocity;
        Wc1 = Wc1 + Wc1_velocity;
        bc1 = bc1 + bc1_velocity;
        
        fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
    end
    alpha = alpha*0.7;
    
    %test accuracy at the end of each Epoch
    %FeedForward Propagation
    %first convolve and pooling
    activations1 = cnnConvolve(filterDim1, numFilters1, testImages, Wc1, bc1);%sigmoid(wx+b)
    activationsPooled1 = cnnPool(poolDim1, activations1);

    %second convolve and pooling
    activations2 = cnnConvolve4D(activationsPooled1, Wc2, bc2);%sigmoid(wx+b)
    activationsPooled2 = cnnPool(poolDim2, activations2);

    activationsPooled2 = reshape(activationsPooled2,[],length(testLabels));
    h = exp(bsxfun(@plus,Wd * activationsPooled2,bd));
    probs = bsxfun(@rdivide,h,sum(h,1)); 
    
    [~,preds] = max(probs,[],1);
    preds = preds';
    acc = sum(preds==testLabels)/length(preds);
    fprintf('Accuracy is %f\n',acc);
end



