function [IT, CR] = GSD_solver(observation, estiS)
 
        tmpM=[];
        for k=0:estiS
        tmpM = [tmpM transpose(observation(1+k:estiS+k)) ];
        end
    candiV = combinator(estiS+1,estiS,'c');
     
    for k=1:size(candiV,1)
         
        selecIndex = candiV(k,:);
        pMat = [];
        for kk=1:length(selecIndex)
            pMat = [pMat tmpM(:,selecIndex(kk))];
        end
        if mod(k,2)~=0
            p(k) = det(pMat);
        else
            p(k) = -det(pMat);
        end
    end
    sol = transpose(roots(p));
        
        Info = [sol];
    
      
    T=[]; TT=[]; tO = (observation);
    for nSol=1:(length(tO))
        T = [T transpose(Info).^(nSol-1)];
    end
    IT = tO*pinv(T);
    CR = Info;
end
