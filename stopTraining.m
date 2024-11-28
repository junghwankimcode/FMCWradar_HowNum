function stop = stopTraining(info,MaxIteration)
NowIteration = info.Iteration;
stop = NowIteration == MaxIteration;
end
