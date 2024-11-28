function [output] = unwrapHankel(m,l)
   
    
    newOb = zeros(1,l);
    cntVec = zeros(1,l);
    for i1=1:size(m,1)
        for i2=1:size(m,2)
            newOb(i1+i2-1) = newOb(i1+i2-1)+m(i1,i2);
            cntVec(i1+i2-1) = cntVec(i1+i2-1)+1;
        end
    end
    
    output = newOb./cntVec;
  
end
