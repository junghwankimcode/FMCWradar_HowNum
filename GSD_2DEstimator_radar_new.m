function [ang] = GSD_2DEstimator_radar_new(Scene, GS, observation, estiS)
 
    preInfo1 = [];
    preInfo2 = [];

    for k=1:size(observation,2)
    p = transpose(observation(:,k)); %all column data 
%     [new_observation] = SVDDENOISING2(p,estiS);
    new_observation = p;
    [tmpInfo1] = GSD_1DangleEstimator(Scene, new_observation, estiS);
    preInfo1 = [preInfo1 ; sort(tmpInfo1,'ComparisonMethod','real')];
    end
%     preInfo1
    for k=1:size(observation,1)
    p = (observation(k,:));
%     [new_observation] = SVDDENOISING2(p,estiS);
    new_observation = p;
    [tmpInfo2] = GSD_1DangleEstimator(Scene, new_observation, estiS);
    preInfo2 = [preInfo2 ; sort(tmpInfo2,'ComparisonMethod','real')]; % sorting using real data
    end
    
    Info1 = (mean(preInfo1));
    Info2 = (mean(preInfo2));
    
    extracted_Info1 = zeros(1,estiS);
    extracted_Info2 = zeros(1,estiS);
    
    %% +(me) mean angles of real target signals
    angles1=cell(1,numel(GS));
    all_angles1=[];
    
    for i = numel(GS)
        angles1{i}=GS{i}.angle1;
        all_angles1 = [all_angles1 angles1{i}];
    end

    mean_angle1=mean(all_angles1);

    angles2=cell(1, numel(GS));
    all_angles2=[];

    for i = numel(GS)
        angles2{i}=GS{i}.angle2;
        all_angles2=[all_angles2 angles2{i}];
    end
    
    mean_angle2=mean(all_angles2);
 
 %% k-trucated SVD   
    if var(Info1) > var(Info2) %column data CR > row data CR
    
    t=[];
    extracted_Info1 = Info1;
    for k=1:size(observation,1)
        p = observation(k,:);
        p = SVDDENOISING2(p,estiS);
        t = [t ; p];
    end
    z = transpose(extracted_Info1).^([0:(size(observation,2))-1]);
    b = t*pinv(z);
    
    for k=1:estiS
    [IT2_resi(k), CR2_resi(k)] = GSD_solver((transpose(b(:,k))), 1);
    end
    
    t=[];
    extracted_Info2 = Info2;
    for k=1:size(observation,2)
        p = transpose(observation(:,k));
        p = SVDDENOISING2(p,estiS);
        t = [t ; p];
    end
    z = transpose(extracted_Info2).^([0:(size(observation,1))-1]);
    b = t*pinv(z);

    for k=1:estiS
    [IT1_resi(k), CR1_resi(k)] = GSD_solver((transpose(b(:,k))), 1);
    end
    
   CR1_sort = zeros(1,estiS);
   for k = 1:estiS
       [ss ii] = min(abs(IT2_resi(k) - IT1_resi));
       CR1_sort(k) = CR1_resi(ii);
       IT1_resi(ii) = Inf;
   end

   CR1 = CR1_sort;
   CR2 = CR2_resi;


   
    else %var(Info1) < var(Info2) %column data CR < row data CR
        
    t=[];
    extracted_Info2 = Info2;
    for k=1:size(observation,2)
        p = transpose(observation(:,k));
        p = SVDDENOISING2(p,estiS);
        t = [t ; p];
    end
    z = transpose(extracted_Info2).^([0:(size(observation,1))-1]);
    b = t*pinv(z);
    
    for k=1:estiS
    [IT1_resi(k), CR1_resi(k)] = GSD_solver((transpose(b(:,k))), 1);
    end
    
    t=[];
    extracted_Info1 = Info1;
    for k=1:size(observation,1)
        p = observation(k,:);
        p = SVDDENOISING2(p,estiS);
        t = [t ; p];
    end
    z = transpose(extracted_Info1).^([0:(size(observation,2))-1]);
    b = t*pinv(z);
    for k=1:estiS
    [IT2_resi(k), CR2_resi(k)] = GSD_solver((transpose(b(:,k))), 1);
    end
   CR2_sort = zeros(1,estiS);
   for k = 1:estiS
       [ss ii] = min(abs(IT1_resi(k) - IT2_resi));
       CR2_sort(k) = CR2_resi(ii);
       IT2_resi(ii) = Inf;
   end

   CR2 = CR2_sort;
   CR1 = CR1_resi;


        
    end

            cpInfo1 = Info1;
        cpexInfo1 = CR1;
        sortInfo1 = zeros(size(preInfo1,1), size(preInfo1,2));
        for k=1:estiS
            [ss ii] = min(abs(cpInfo1 - cpexInfo1(k)));
            sortInfo1(:,k) = preInfo1(:,ii);
            cpInfo1(ii) = Inf;
        end
        CR1 = [sortInfo1 ; CR1];
        CR1 = mean(CR1);
        
                cpInfo2 = Info2;
        cpexInfo2 = CR2;
        sortInfo2 = zeros(size(preInfo2,1), size(preInfo2,2));
        for k=1:estiS
            [ss ii] = min(abs(cpInfo2 - cpexInfo2(k)));
            sortInfo2(:,k) = preInfo2(:,ii);
            cpInfo2(ii) = Inf;
        end
        CR2 = [sortInfo2 ; CR2];
        CR2 = mean(CR2);
    
   %final pairing
   tmpAng(1,:) = CR1;
   tmpAng(2,:) = CR2; 

%%
   if estiS <= size(GS,2) %size(GS,2)=testDoF
    ang = cell(1,estiS);

    for k=1:estiS
        
        [ss ii] = min(abs(GS{k}.angle1 - tmpAng(1,:)));      % tmpAng= estimation delay?
        ang{k}.angle1 = tmpAng(1,ii) / abs(tmpAng(1,ii));
        tmpAng(1,ii) = Inf;
        
        [ss ii] = min(abs(GS{k}.angle2 - tmpAng(2,:)));       % tmpAng= estimation doppler?
        ang{k}.angle2 = tmpAng(2,ii) / abs(tmpAng(2,ii)); 
        tmpAng(2,ii) = Inf;
        
    end

   % elseif estiS < size(GS,2) 
   %      ang = cell(1,estiS);
   %      for i =1:numel(ang)
   %          ang{i}.angle1=mean_angle1;
   %          ang{i}.angle2=mean_angle2;
   %      end
   % 
   %     for k=1:estiS
   % 
   %      [ss ii] = min(abs(GS{k}.angle1 - tmpAng(1,:)));       
   %      ang{k}.angle1 = tmpAng(1,ii) / abs(tmpAng(1,ii));
   %      tmpAng(1,ii) = Inf;
   % 
   %      [ss ii] = min(abs(GS{k}.angle2 - tmpAng(2,:)));       
   %      ang{k}.angle2 = tmpAng(2,ii) / abs(tmpAng(2,ii));
   %      tmpAng(2,ii) = Inf;
   %     end
      


   else % estiS > size(GS,2) 
     ang = cell(1,size(GS,2));
        % for i =1:numel(ang)
        %     ang{i}.angle1=mean_angle1;
        %     ang{i}.angle2=mean_angle2;
        % end
        % 

     for k=1:size(GS,2) 

        [ss ii] = min(abs(GS{k}.angle1 - tmpAng(1,:)));
        ang{k}.angle1 = tmpAng(1,ii) / abs(tmpAng(1,ii));
        tmpAng(1,ii) = Inf;
        
        [ss ii] = min(abs(GS{k}.angle2 - tmpAng(2,:)));       
        ang{k}.angle2 = tmpAng(2,ii) / abs(tmpAng(2,ii));
        tmpAng(2,ii) = Inf;

     end

   end

end
