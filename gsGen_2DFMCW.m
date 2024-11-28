function [GS, superposition, observation] = gsGen_2DFMCW(Scene, SNR, S, P1, P2, gain, D, V, c, f_c, f_s, Tsweep, K)
    GS = cell(1,S); % Init. S Sequence
    n_spacing = 1; % not need 

    m_spacing = 1; %not need

    %D=2R
switch(Scene)
        case 2
%             disp('Complex Ring Domain')

            for ii=1:S
                GS{ii}.gain = gain(ii); 
                GS{ii}.Doppler = (2 * f_c)* V(ii) /c;   
                
                GS{ii}.IT = GS{ii}.gain*exp(i*2*pi*f_c*D(ii)/c);
                
%                 GS{ii}.rvInfo = ((K*D(ii)/c) + GS{ii}.Doppler) / f_s;
                GS{ii}.rvInfo = ((K*D(ii)/c)) / f_s; % approximate 
                GS{ii}.vInfo = GS{ii}.Doppler*Tsweep;
                GS{ii}.angle1 = exp(i*2*pi*GS{ii}.rvInfo);
                GS{ii}.angle2 = exp(i*2*pi*GS{ii}.vInfo);
                
                for iii=1:P1
                    for iiii=1:P2
                    GS{ii}.Form(iii,iiii) = GS{ii}.gain * exp(i*2*pi*n_spacing*(GS{ii}.rvInfo))^(iii-1) * exp(i*2*pi*m_spacing*(GS{ii}.vInfo))^(iiii-1);
                    end
                end
            end
        case 3
            disp('Complex Domain')

end
 
    superposition=zeros(P1,P2);
    for ii=1:S
        superposition = superposition + (GS{ii}.Form);
    end
    observation = awgn(superposition(1:P1,1:P2), SNR, 'measured');

    
end
