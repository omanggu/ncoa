function [xbest,fbest,FES] = ncoa(fcn_handle,dim,lb,ub,max_FES,accuracy,pop_size,rngstate)
    %NCOA - Niching Chaos Optimization Algorithm
    %            to solve multimodal optimization problems
    % [xbest,fbest,FES] = ancoa(fcn_handle,dim,lb,ub,max_FES,rngstate)
    %
    % input variables
    %  fun_handle : handle of objective function
    %         dim : dimension of objective function
    %          lb : lower boundaries
    %          ub : upper boundaries
    %     max_FES : limit number of function evaluations
    %    accuracy : accuracy of solution
    %    pop_size : size of population
    % is_maximize : whether maximization problem or not
    %    rngstate : state of the random number generator
    %
    % output variables
    %       xbest : global/local optima found
    %       fbest : evaluated function value on found solution
    %         FES : number of evaluations
    %
    % Last Modified on Oct. 18, 2016
    % Cholmin Rim
    % pyonghung@163.com
    % KIM IL SUNG University, D.P.R.Korea

    %% process of input variables
    switch nargin
        case 0
            fcn_handle = @(x)(sum((x-0.1982032019840504).^2));
            dim = 10; 
            lb = 0.0; ub = 1.0;
            max_FES = 10000; 
            accuracy = 1e-3;
            pop_size = 10;
            rngstate = sum(100*clock);
        case 1
            dim = 1; 
            lb = 0.0; ub = 1.0;
            max_FES = 10000; 
            accuracy = 1e-3;
            pop_size = 10;
            rngstate = sum(100*clock);
        case 2
            lb = 0.0; ub = 1.0;
            max_FES = 10000; 
            accuracy = 1e-3;
            pop_size = 10;
            rngstate = sum(100*clock);
        case 4
            max_FES = 10000; 
            accuracy = 1e-3;
            pop_size = 10;
            rngstate = sum(100*clock);
        case 5
            accuracy = 1e-3;
            pop_size = 10;
            rngstate = sum(100*clock);
        case 6
            pop_size = 10;
            rngstate = sum(100*clock);
        case 7
            rngstate = sum(100*clock);
        otherwise
            error('Input parameters are incorrect.')
    end
    rand('state',rngstate);
    if length(lb)==1
        lb = repmat(lb,dim,1); ub = repmat(ub,dim,1);
    end
    band = ub - lb;

    %% initialize
%     xbest = repmat(lb,1,pop_size) + rand(dim,pop_size).*repmat(band,1,pop_size);
    xbest = initPop(pop_size,dim,lb,ub);
    fbest = zeros(1,pop_size);
    for i=1:pop_size
        fbest(i) = feval(fcn_handle,xbest(:,i));
    end
    FES = pop_size;

    % contraction parameters
    steps = 50;
    num_sample = max( 5+ceil(dim/20),  floor(max_FES/(steps*pop_size)) );
    contract_ratio = max( 0.5, accuracy^(pop_size*num_sample/max_FES) );
    fai = 0.5;
    % search scopes
    mutative_lb = max( repmat(lb,1,pop_size), xbest - fai.*repmat(band,1,pop_size) );
    mutative_ub = min( repmat(ub,1,pop_size), xbest + fai.*repmat(band,1,pop_size) );
    clear_radius = 0.01*dim;         % clearing radius
    index = 1;                       % index of individual
    z = rand(dim,pop_size);          % chaos variavles
    for i=1:100, 
        z = circleMap(z); 
    end
    FES_in_stage = 0;

    %% steps
    while (FES<max_FES)
        if FES_in_stage >= pop_size*num_sample
            FES
            %% clearing
            clear_mat = zeros(pop_size-1,pop_size);
            for i=1:(pop_size-1)
                for j=(i+1):pop_size
                    dist = norm( (xbest(:,i)-xbest(:,j))./band );
                    clear_mat(i,j) = ( dist < clear_radius );
                end
            end
            num_cleared = sum(sum(clear_mat));
            if num_cleared>0
                cleared_members = [];
                [i,j] = find(clear_mat>0);
                for k=1:num_cleared
                    if fbest(i(k))>fbest(j(k))
                        cleared_members = [cleared_members,i(k)];
                    else
                        cleared_members = [cleared_members,j(k)];
                    end
                end
                revived_members = [];
                for k=1:pop_size
                    if sum(k==cleared_members)==0
                        revived_members = [revived_members,k];
                    end
                end
                xbest = xbest(:,revived_members);
                fbest = fbest(:,revived_members);
                z = z(:,revived_members);

                pop_size = length(revived_members);
                index = ceil(rand*pop_size);
            end

            %% contracting
            fai = fai*contract_ratio;
            mutative_lb = max( repmat(lb,1,pop_size), xbest - fai*repmat(band,1,pop_size) );
            mutative_ub = min( repmat(ub,1,pop_size), xbest + fai*repmat(band,1,pop_size) );
            FES_in_stage = 0;
        end

        x0 = mutative_lb(:,index) + z(:,index).*(mutative_ub(:,index)-mutative_lb(:,index));
        f0 = feval(fcn_handle,x0); FES = FES+1;
        FES_in_stage = FES_in_stage+1;

        dist = sum( ( (xbest-x0*ones(1,pop_size))./repmat(band,1,pop_size) ).^2,1);  % 2-norm distance, root is omited
        [~,neighbor] = min(dist);

        if f0<fbest(neighbor)
            xbest(:,neighbor) = x0;
            fbest(neighbor) = f0;
        end

        index = mod(index,pop_size)+1;
        z(:,index) = circleMap(z(:,index));
    end

    % process the results
    [fbest,sorted] = sort(fbest);
    xbest = xbest(:,sorted);
end

%% Circle map
function z = circleMap(z)
    % z : chaos variable in [0,1]
    % z(k+1) = z(k) + omega - (K/(2*pi))*sin(2*pi*z(k)) mod 1
    % omega = 0.5; K = 11;
    z = mod(z+0.5-11./(2*pi)*sin(2*pi*z),1);
end

%% initialize
function initx = initPop(popsize,nvars,lb,ub)
    % Initialize the poplation
    if length(lb)==1
        lb=repmat(lb,nvars,1);
        ub=repmat(ub,nvars,1);
    end
    p = ceil(popsize^(1/nvars));
    patitions = p^nvars;
    initx = zeros(nvars,popsize);
    index = floor(rand*patitions);
    for i=1:popsize
        for j=1:nvars
            k = mod(floor(index/p^(j-1)),p);
            initx(j,i) = lb(j) + (ub(j)-lb(j))/p*(k+0.5);
        end
        index = mod(index+p+1,patitions);
    end
end


