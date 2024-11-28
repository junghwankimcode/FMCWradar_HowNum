function [output,iterVal] = SVDDENOISING2(observation,S)
    iterMax = 30;
    iterVal = iterMax;
    
    l = length(observation);
    
    r = ceil(l/2);
    c = l - r + 1;
    m = zeros(r,c);
    
    for iter=1:iterMax
    
    for k=1:r
        m(k,:) = (observation(k:k+c-1));
    end
    
    [u s v] = svd(m);
    
    m = u(:,[1:S])*(s(1:S,1:S))*v(:,[1:S])';
    
%     for k=1:c
%        observation(k) = m(1,k); 
%     end
%     for k=c+1:l
%         observation(k)=m(k-c+1,c) ;
%     end
%     

newOb = zeros(1,l);
cntVec = zeros(1,l);
    for i1=1:size(m,1)
        for i2=1:size(m,2)
            newOb(i1+i2-1) = newOb(i1+i2-1)+m(i1,i2);
            cntVec(i1+i2-1) = cntVec(i1+i2-1)+1;
        end
    end
    
    if norm(observation - newOb./cntVec) < 1e-10
%         disp('svd Converge [1]')
        iterVal = iter;
        break
    else
    observation = newOb./cntVec;
    end
    end
output = observation;
    
end
