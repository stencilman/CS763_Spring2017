require 'torch'
-- load trainin images
tr_x = torch.load('cifar10/tr_data.bin')
-- load trainin labels 
tr_y = torch.load('cifar10/tr_labels.bin'):double() + 1
-- load test images
te_x = torch.load('cifar10/te_data.bin')
-- load test labels 
te_y = torch.load('cifar10/te_labels.bin'):double() + 1
print(tr_x:size())
print(tr_y:size())

-- display the first 36 training set images
require 'image';
itorch.image(tr_x[{{1,36},{},{},{}}])

x_mean = torch.mean(tr_x:float(), 1)
x_std = torch.std(tr_x:float(), 1)
itorch.image(x_mean)
itorch.image(x_std)



-- params: dataset (test, train) and index in that
function get_xi(data_x, idx)   
    xi = (data_x[idx]:float() - x_mean)
    xi = xi:cdiv(x_std)
--     xi = xi:reshape(3*32*32)
    return xi
end

function mod(a, b)
    return a - math.floor(a/b)*b
end

require 'math'
randomIdx = {}
for i = 1,100 do
    table.insert(randomIdx, math.random(10000))
end
function evaluate(model, data_x, data_y) 
    errors = 0
    for i = 1,#randomIdx do
        idx = randomIdx[i]
        xi = get_xi(data_x, idx)
        op = model:forward(xi)
        _, op_label = torch.max(op, 1)
        ti = data_y[idx]
        if ti ~= op_label[1] then
            errors = errors + 1
        end
    end
    return errors/#randomIdx
end

require 'nn'
function createModel()
    model = nn.Sequential()
    model = model:add(nn.SpatialConvolutionMM(3,16,5,5))
    return model
end

model = createModel()
criterion = nn.CrossEntropyCriterion()

function train_and_test_loop(no_iterations, lr, lambda)
    for i = 0, no_iterations do
        -- shuffle data
        if mod(i, tr_x:size(1)) == 0 then
            shuffle = torch.randperm(tr_x:size(1))
        end
        if mod(i, te_x:size(1)) == 0 then
            shuffle_te = torch.randperm(te_x:size(1))
        end

        -- learning rate multiplier
        if i == 60000 then lr = lr * 0.1 end

        -- trainin input and target
        idx = shuffle[mod(i, tr_x:size(1)) + 1] 
        xi = get_xi(tr_x, idx) 
        ti = tr_y[idx]
        -- do forward of the model, compute loss
        -- and then do backward of the model
        op = model:forward(xi)
        loss_tr = criterion:forward(op, ti)
        dl_do = criterion:backward(op, ti)
        model:backward(xi, dl_do)
        epochloss_tr = epochloss_tr + loss_tr
        
        -- udapte model weights
        model:updateParameters(lr)
        model:zeroGradParameters()

        -- test input and target
        idx = shuffle_te[mod(i, te_x:size(1)) + 1] 
        xi = get_xi(te_x, idx) 
        ti = te_y[idx]
        -- do forward of the model and compute loss
        op = model:forward(xi) 
        loss_te = criterion:forward(op, ti, model, lambda)
        epochloss_te = epochloss_te + loss_te

        

        if mod(i, 1000) == 0 then
            if i ~= 0 then
                table.insert(epochlosses_te, epochloss_te/1000)
                table.insert(epochlosses_tr, epochloss_tr/1000)
            end
            epochloss_te = 0
            epochloss_tr = 0
            err = evaluate(model, tr_x, tr_y)
            print('iter: '..i.. ', accuracy: '..(1 - err)*100 ..'%')
            if (err < besterr) then
                besterr = err
                bestmodel:copy(model)
                print(' -- best accuracy achieved: '.. (1- besterr)*100 ..'%')
            end
            collectgarbage()
        end
    end
end

besterr = 1e10

-- for plotting losses later on
epochloss_te = 0
epochloss_tr = 0
epochlosses_tr = {}
epochlosses_te = {}

-- run it
lr = 0.00001
lambda = 0.0
train_and_test_loop(100000, lr, lambda)

Plot = require 'itorch.Plot'
xaxis = {}
for i=1, #epochlosses_tr do
    table.insert(xaxis, i)
end
plot = Plot():line(xaxis, epochlosses_tr, 'red', 'train')
plot:line(xaxis, epochlosses_te, 'green', 'test'):legend(true):title('Train and Test Loss')
plot:draw():save('out.html')

Ws = {}
for i = 1, 10 do
    table.insert(Ws, bestmodel.W[i]:reshape(3,32,32))
end
itorch.image(Ws)

besterr = 1e10

-- for plotting losses later on
epochloss_te = 0
epochloss_tr = 0
epochlosses_tr = {}
epochlosses_te = {}

-- define the model and criterion
model = Linear.new(0.0001)
criterion = CEC.new()
bestmodel = Linear.new(0)

-- run it
lr = 0.0001
lambda = 0.0
train_and_test_loop(100000, lr, lambda)

Plot = require 'itorch.Plot'
xaxis = {}
for i=1, #epochlosses_tr do
    table.insert(xaxis, i)
end

plot = Plot():line(xaxis, epochlosses_tr, 'red', 'train')
plot:line(xaxis, epochlosses_te, 'green', 'test'):legend(true):title('Train and Test Loss')
plot:draw():save('out.html')

Ws = {}
for i = 1, 10 do
    table.insert(Ws, bestmodel.W[i]:reshape(3,32,32))
end
itorch.image(Ws)

besterr = 1e10

-- for plotting losses later on
epochloss_te = 0
epochloss_tr = 0
epochlosses_tr = {}
epochlosses_te = {}

-- define the model and criterion
model = Linear.new(0.01)
criterion = CEC.new()
bestmodel = Linear.new(0)

-- run it
lr = 0.0001
lambda = 0.1
train_and_test_loop(100000, lr, lambda)

Plot = require 'itorch.Plot'
xaxis = {}
for i=1, #epochlosses_tr do
    table.insert(xaxis, i)
end
plot = Plot():line(xaxis, epochlosses_tr, 'red', 'train')
plot:line(xaxis, epochlosses_te, 'green', 'test'):legend(true):title('Train and Test Loss')
plot:draw():save('out.html')

Ws = {}
for i = 1, 10 do
    table.insert(Ws, bestmodel.W[i]:reshape(3,32,32))
end
itorch.image(Ws)
