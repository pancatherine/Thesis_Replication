%% introduction
% 
% Paper reproduction
%               @By Yanting
clear;
close all;

function plot_UAV_connections(uav_positions, bs_positions, ue_positions, c_U, c_F, area_size)
    % Plot positions and connections in the UAV network
    % 
    % Inputs:
    %   uav_positions: (N x 2) matrix of UAV coordinates
    %   bs_positions: (F x 2) matrix of BS coordinates
    %   ue_positions: (M x 2) matrix of UE coordinates
    %   c_U: (N x N) matrix for UAV-to-UAV connections (1 for active, 0 for inactive)
    %   c_F: (F x N) matrix for BS-to-UAV connections (1 for active, 0 for inactive)
    %   b_U: (N x M) matrix for UAV-to-UE connections (1 for active, 0 for inactive)
    %   b_F: (F x M) matrix for BS-to-UE connections (1 for active, 0 for inactive)
    %   area_size: Scalar defining the area size for plot boundaries
    
    % Set up the plot
    figure;
    hold on;
    axis([0 area_size 0 area_size]);
    title('UAV-BS UAV-UAV Network Connections');
    xlabel('X Position');
    ylabel('Y Position');
    
    % Plot base stations
    scatter(bs_positions(:, 1), bs_positions(:, 2), 100, 's', 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b');
    text(bs_positions(:, 1), bs_positions(:, 2), 'BS', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    
    % Plot UAVs
    scatter(uav_positions(:, 1), uav_positions(:, 2), 80, 'd', 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g');
    text(uav_positions(:, 1), uav_positions(:, 2), 'UAV', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    
    % Plot UEs
    scatter(ue_positions(:, 1), ue_positions(:, 2), 50, '^', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r');
    text(ue_positions(:, 1), ue_positions(:, 2), 'UE', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    
    % Plot UAV-to-UAV connections
    N = size(uav_positions, 1);
    for i = 1:N
        for j = i+1:N
            if c_U(i, j) == 1
                plot([uav_positions(i, 1), uav_positions(j, 1)], [uav_positions(i, 2), uav_positions(j, 2)], 'r-', 'LineWidth', 1); % Active UAV-to-UAV connection
               
            else
                plot([uav_positions(i, 1), uav_positions(j, 1)], [uav_positions(i, 2), uav_positions(j, 2)], 'r--', 'LineWidth', 1); % Inactive UAV-to-UAV connection
     
            end
        end
    end
    
    % Plot BS-to-UAV connections
    F = size(bs_positions, 1);
    for i = 1:F
        for j = 1:N
            if c_F(i, j) == 1
                plot([bs_positions(i, 1), uav_positions(j, 1)], [bs_positions(i, 2), uav_positions(j, 2)], 'm-', 'LineWidth', 1); % Active BS-to-UAV connection
            else
                plot([bs_positions(i, 1), uav_positions(j, 1)], [bs_positions(i, 2), uav_positions(j, 2)], 'm--', 'LineWidth', 1); % Inactive BS-to-UAV connection
            end
        end
    end
    
    
    % % Legend
    % legend({'Base Stations', 'UAVs', 'UEs', 'Active UAV-to-UAV', 'Inactive UAV-to-UAV', 'Active BS-to-UAV', 'Inactive BS-to-UAV'}, ...
    %        'Location', 'bestoutside');
    
    hold off;
end



function plot_UE_network_connections(uav_positions, bs_positions, ue_positions, b_U, b_F, area_size)
    % Plot positions and connections in the UAV network
    % 
    % Inputs:
    %   uav_positions: (N x 2) matrix of UAV coordinates
    %   bs_positions: (F x 2) matrix of BS coordinates
    %   ue_positions: (M x 2) matrix of UE coordinates
    %   c_U: (N x N) matrix for UAV-to-UAV connections (1 for active, 0 for inactive)
    %   c_F: (F x N) matrix for BS-to-UAV connections (1 for active, 0 for inactive)
    %   b_U: (N x M) matrix for UAV-to-UE connections (1 for active, 0 for inactive)
    %   b_F: (F x M) matrix for BS-to-UE connections (1 for active, 0 for inactive)
    %   area_size: Scalar defining the area size for plot boundaries
    
    % Set up the plot
    figure;
    hold on;
    axis([0 area_size 0 area_size]);
    title('BS-UE UAV-UE Network Connections');
    xlabel('X Position');
    ylabel('Y Position');
    
    % Plot base stations
    scatter(bs_positions(:, 1), bs_positions(:, 2), 100, 's', 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'b');
    text(bs_positions(:, 1), bs_positions(:, 2), 'BS', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    
    % Plot UAVs
    scatter(uav_positions(:, 1), uav_positions(:, 2), 80, 'd', 'filled', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'g');
    text(uav_positions(:, 1), uav_positions(:, 2), 'UAV', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    
    % Plot UEs
    scatter(ue_positions(:, 1), ue_positions(:, 2), 50, '^', 'MarkerEdgeColor', 'k', 'MarkerFaceColor', 'r');
    text(ue_positions(:, 1), ue_positions(:, 2), 'UE', 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
    
    
    % Plot BS-to-UE connections
    M = size(ue_positions, 1);
    F = size(bs_positions, 1);
    N = size(uav_positions, 1);
    for i = 1:F
        for k = 1:M
            if b_F(i, k) == 1
                plot([bs_positions(i, 1), ue_positions(k, 1)], [bs_positions(i, 2), ue_positions(k, 2)], 'b-', 'LineWidth', 1.5); % Active BS-to-UE connection
            end
        end
    end
    
    % Plot UAV-to-UE connections
    for j = 1:N
        for k = 1:M
            if b_U(j, k) == 1
                plot([uav_positions(j, 1), ue_positions(k, 1)], [uav_positions(j, 2), ue_positions(k, 2)], 'g-', 'LineWidth', 1.0); % Active UAV-to-UE connection
            end
        end
    end
    
    % % Legend
    % legend({'Base Stations', 'UAVs', 'UEs', 'Active BS-to-UE', 'Active UAV-to-UE'}, ...
    %        'Location', 'bestoutside');
    % 
    hold off;
end




% Algorithm 1 Initialization
% Random positions for demonstration (assuming a 2D plane for simplicity)
% uav_positions = rand(N, 2) * 200; % Positions of UAVs (e.g., in a 500x500 area)
% bs_positions = rand(F, 2) * 200;  % Positions of BSs
% ue_positions = rand(M, 2) * 200;  % Positions of UEs

bs_positions = [40,0;60,0]; % Positions of BSs
uav_positions = [20,30;50,30;80,30]; % Positions of UAVs (e.g., in a 500x500 area)
ue_positions =[0,100; 20,100;50,100;80,100;100,100];  % Positions of UEs


% Parameters
% N = 2; % Number of UAVs (example)
% F = 2;  % Number of BSs (example)
% M = 3; % Number of UEs (example)
N = size(uav_positions, 1);
F = size(bs_positions, 1);
M = size(ue_positions, 1);
% N = 6; % Number of UAVs (example)
% F = 2;  % Number of BSs (example)
% M = 4; % Number of UEs (example)
Rc = 100; % Communication range threshold
Rt = 50; % Maximum coverage radius for UAVs
sqrt3Rt = sqrt(3) * Rt;


% Initialize variables
a = ones(N, 1);        % State of each UAV (1 = active, 0 = idle)


% Initialize connection matrices
c_U = zeros(N, N);     % Single-hop connection matrix among UAVs
c_F = zeros(F, N);     % Single-hop connection matrix between BSs and UAVs



% Calculate distances and populate connection matrices
for i = 1:N

     % Distance between UAV i and UAV j
    for j = 1:N
        if i ~= j
            dist_UU = norm(uav_positions(i, :) - uav_positions(j, :)); % Distance between UAV i and UAV j
            if dist_UU <= Rc
                c_U(i, j) = 1; % Set connection if within communication range
            end
        end
    end

    % Distance between BS l and UAV i
    for l = 1:F
        dist_FB = norm(bs_positions(l, :) - uav_positions(i, :)); % Distance between BS l and UAV i
        if dist_FB <= Rc
            c_F(l, i) = 1; % Set connection if within communication range
        end
    end
end

% Plotting
area_size = 200;
plot_UAV_connections(uav_positions, bs_positions, ue_positions, c_U, c_F, area_size)





%
% Algorithm 2 Build/Update Connection Graphs

% Parameters

% Const for pathloss
f0 = 2e9; % Carrier frequency in Hz (example: 2 GHz)
c = 3e8; % Speed of light in m/s

% Const for connection
SINR_threshold = 0.2;  % SINR threshold for connection
ph = 0.1;             % Transmission power for UAVs
pf = 0.2;             % Transmission power for BSs
N0 = 1e-3;            % Noise power

alpha = 1; % ??
beta = 2; % ??
eta_LoS = 0.1; % ??
eta_NLoS = 0.2; % ??

height_UAV = 10; 
height_BS = 5; 

% Association matrices and degrees from Algorithm 1 (assumed initialized)
b_U = zeros(N, M);    % Association between UAVs and UEs
b_F = zeros(F, M);    % Association between BSs and UEs

zeta_u = zeros(N, 1); % Degree of each UAV (number of UEs connected to each UAV)
zeta_f = zeros(F, 1); % Degree of each BS (number of UEs connected to each BS)

xi_u = zeros(M, 1);   % Degree of each UE - UAVs
xi_f = zeros(M, 1);   % Degree of each UE - BSs
xi = zeros(M, 1);     % Total degree of each UE

% Distances between UAVs and UEs, BSs and UEs (example calculation)
uav_ue_distances = pdist2(uav_positions, ue_positions); % N x M matrix
bs_ue_distances = pdist2(bs_positions, ue_positions);   % F x M matrix

% Calculate the elevation angle in radians
tan_uav_ue = height_UAV ./ uav_ue_distances; % Angle between UAV i and point k (in degrees or radians)
theta_uav_ue = atan(tan_uav_ue);

tan_bs_ue = height_BS ./ bs_ue_distances; % Angle between UAV i and point k (in degrees or radians)
theta_bs_ue = atan(tan_bs_ue);



% Pathloss BS - UE
P_LoS_bs_ue = 1 ./ (1 + alpha * exp(-beta * (theta_bs_ue - alpha))); % matrix
FSPL_bs_ue = 20 * log10(bs_ue_distances) + 20 * log10(f0) + 20 * log10(4 * pi / c); % matrix
average_pathloss_LoSbs_ue = FSPL_bs_ue + eta_LoS ; % matrix
average_pathloss_NLoSbs_ue = FSPL_bs_ue + eta_NLoS ; % matrix
g_bs_ue = P_LoS_bs_ue .* average_pathloss_LoSbs_ue + (1 - P_LoS_bs_ue) .* average_pathloss_NLoSbs_ue; % pathloss - matrix


% Pathloss UAV - UE
P_LoS_uav_ue = 1 ./ (1 + alpha * exp(-beta * (theta_uav_ue - alpha))); % matrix
FSPL_uav_ue = 20 * log10(uav_ue_distances) + 20 * log10(f0) + 20 * log10(4 * pi / c);  % matrix
average_pathloss_LoSuav_ue = FSPL_uav_ue + eta_LoS ; % matrix
average_pathloss_NLoSuav_ue = FSPL_uav_ue + eta_NLoS ;  % matrix
g_uav_ue = P_LoS_uav_ue .* average_pathloss_LoSuav_ue + (1 - P_LoS_uav_ue) .* average_pathloss_NLoSuav_ue; % pathloss - matrix


% Loop through each BS, UAV, and UE to calculate SINR and update connections
for i = 1:F
    for k = 1:M
        % Calculate SINR for BS i and UE k
        % Calculate the path loss for each pair 
        signal_power = pf * g_bs_ue(i,k);

        % Calculate interference BS UE (sum of other BSs and UAVs)
        interference = N0; % Start with noise power
        for j = [1:i-1, i+1:F]  % Other BSs
            interference = interference + pf * g_bs_ue(j,k);
        end
        
        % Calculate SINR and update b_F, zeta_f, xi_f if connection is valid
        SINR_BS_UE = signal_power / interference;

        if SINR_BS_UE >= SINR_threshold
            b_F(i, k) = 1;
            zeta_f(i) = zeta_f(i) + 1;
            xi_f(k) = xi_f(k) + 1;
            xi(k) = xi(k) + 1;
        end
    end
end

for j = 1:N
    for k = 1:M
        % Calculate SINR for UAV j and UE k
        signal_power = ph * g_uav_ue(j,k);

        % Calculate interference (sum of other UAVs and BSs)
        interference = N0; % Start with noise power
        for n = [1:j-1, j+1:N]  % UAVs
            interference = interference + ph * g_uav_ue(n, k);
        end
        
        % Calculate SINR and update b_U, zeta_u, xi_u if connection is valid
        SINR_UAV_UE = signal_power / interference;
        if SINR_UAV_UE >= SINR_threshold
            b_U(j, k) = 1;
            zeta_u(j) = zeta_u(j) + 1;
            xi_u(k) = xi_u(k) + 1;
            xi(k) = xi(k) + 1;
        end
    end
end

plot_UE_network_connections(uav_positions, bs_positions, ue_positions, b_U, b_F, area_size)

%
% Algorithm 3 Delete Redundant Connections Between BSs/UAVs and UEs

% Parameters (from previous algorithms)
Mmax_F = 2; % Maximum number of UEs per BS
Mmax_H = 2;  % Maximum number of UEs per UAV

% Inputs from Algorithm 2 (for demonstration, assumed initialized)
% b_F, b_U, zeta_f, zeta_u, xi_f, xi_u, xi (association matrices and degrees)

% Loop to remove redundant connections while there are UEs connected to more than one node


while max(xi) > 1
    % Step 1: Handle Redundant Connections for BSs
    % cnt=1;
    while max(xi_f) > 1 %|| length( find(b_F(i, :) == 1)) > Mmax_F % Check if any UE is connected to multiple BSs
        % cnt=cnt+1;
        % disp(cnt);
       
        [~, i] = max(zeta_f);% Find BS with the highest degree
    
        % Find UEs connected to this BS
        phi = find(b_F(i, :) == 1);  % Set of UEs connected to BS i
        psi = xi_f(phi);             % Degrees of these UEs to BSs

        % Remove extra connections if BS degree exceeds Mmax_F
        while length(phi) > Mmax_F
            % Find the UE with the highest degree among the connected UEs
            [~, k_idx] = max(psi);
            k = phi(k_idx); 
            % UE-BS important
            b_F(i, k) = 0;  % Remove connection from BS i to UE k
            zeta_f(i) = zeta_f(i) - 1;  % Decrease BS degree
            xi_f(k) = xi_f(k) - 1;      % Decrease UE-BS degree
            xi(k) = xi(k) - 1;          % Decrease total degree of UE
            % Update variables after deletion
            phi(k_idx) = [];
            psi(k_idx) = [];
            %
    
        end
    
    
        % Remove all other connections to these UEs from different BSs
        b_F(i, :) = 0;
        for k = phi
            b_F(:, k) = 0;  % Clear connections to other BSs for UE k
            b_F(i, k) = 1;  % Retain only the connection to the current BS i
            b_U(:, k) = 0; % ok

        end


    while max(sum(b_F, 2))> Mmax_F
        zataf = sum(b_F, 2)';
        problem_BS_indices = find(zataf > Mmax_F);
        for BS = problem_BS_indices
            zataf = sum(b_F, 2)';
            zatau = sum(b_U, 2)';
        
            xif = sum(b_F, 1);
            xiu = sum(b_U, 1);
        
            ue_BSi = b_F(BS,:);
            ue_Indices = find(ue_BSi ~= 0);
            degree_ue = xif(ue_Indices) + xiu(ue_Indices);
        
            
            [~, max_ue_indice] = max(degree_ue);
            original_ue_index = ue_Indices(max_ue_indice);
            
            b_F(BS,original_ue_index) = 0;
            
            % degree_f = sum(b_F, 2)'
        
        end
    end

        xi_f = sum(b_F,1)'; % ok   
        xi_u = sum(b_U,1)'; % ok
        zeta_f = sum(b_F, 2);  
        zeta_u = sum(b_U, 2);  
        xi = xi_f + xi_u; % ok
    end
    disp('Step 1 finished.');
    
    % Step 2: Handle Redundant Connections for UAVs
    while max(xi_u) > 1 %|| length( find(b_U(j, :) == 1)) > Mmax_H % Check if any UE is connected to multiple UAVs
        xi_u_copy = xi_u;
        [~, j] = max(zeta_u); % Find UAV with the highest degree
    
        % Find UEs connected to this UAV
        phi_prime = find(b_U(j, :) == 1);  % Set of UEs connected to UAV j
        psi_prime = xi_u(phi_prime);       % Degrees of these UEs to UAVs
        
        % Remove extra connections if UAV degree exceeds Mmax_H
        if length(phi_prime) > Mmax_H
            while length(phi_prime) > Mmax_H
                % Find the UE with the highest degree among the connected UEs
                [~, l_idx] = max(psi_prime);
                l = phi_prime(l_idx);
        
                b_U(j, l) = 0;  % Remove connection from UAV j to UE l
                zeta_u(j) = zeta_u(j) - 1;  % Decrease UAV degree
                xi_u(l) = xi_u(l) - 1;      % Decrease UE-UAV degree
                xi(l) = xi(l) - 1;          % Decrease total degree of UE
        
                % Update variables after deletion
                phi_prime(l_idx) = [];
                psi_prime(l_idx) = [];
        
            end
        else
                     
            % no change, manually delete the extra link    
            if xi_u_copy == xi_u
                v = b_U(:,j);
                nonZeroIndices = find(v ~= 0);
                nonZeroIndices = nonZeroIndices';
                zeta_u = sum(b_U, 2)';
                % 获取这些索引对应的值
                values = zeta_u(nonZeroIndices);
                
                % 找到最大值的索引
                [~, maxIdx] = max(values);
                % 对应的原始索引
                maxIndexInOriginal = nonZeroIndices(maxIdx);
                % 保maxIndexInOriginal，删其他的
    
                nonZeroIndices(nonZeroIndices == maxIndexInOriginal) = [];
     
                b_U(:, j) = 0;
                b_U(maxIndexInOriginal, j) = 1;
    
                phi_prime = find(b_U(j, :) == 1);  % Set of UEs connected to UAV j
                psi_prime = xi_u(phi_prime);       % Degrees of these UEs to UAVs
    
    
            end
        end
        % Remove all other connections to these UEs from different UAVs
        b_U(j, :) = 0;
        for l = phi_prime
            b_U(:, l) = 0;  % Clear connections to other UAVs for UE l
            b_U(j, l) = 1;  % Retain only the connection to the current UAV j
        end
    
        xi_u = sum(b_U)'; % ok
        zeta_u = sum(b_U, 2);  
        xi = xi_f + xi_u; % ok
    
    end
    
    % Step 3: Final Check for UEs with Multiple Connections
    if max(xi) > 1  % There are still UEs connected to both BSs and UAVs
        phi_double = find(xi > 1);  % UEs with multiple connections
        % Remove UAV connections for these UEs
        for k = phi_double
            b_U(:, k) = 0;  % Set all UAV connections for UE k to 0
        end
    end
    
    % Update matrices and degrees after removal of redundant connections
    zeta_f = sum(b_F, 2);  % Recalculate BS degrees
    zeta_u = sum(b_U, 2);  % Recalculate UAV degrees
    xi_f = sum(b_F, 1)';   % Recalculate UE degrees for BS connections
    xi_u = sum(b_U, 1)';   % Recalculate UE degrees for UAV connections
    xi = xi_f + xi_u;      % Recalculate total degree of each UE
end


plot_UE_network_connections(uav_positions, bs_positions, ue_positions, b_U, b_F, area_size)








% Algorithm 4 Check Network Bi-Connectivity Upon Deletion of Idle UAV

% Inputs from the previous algorithms (assumed initialized)
% N: Total number of UAVs
% c_U: Connectivity matrix among UAVs (N x N)
% a: State vector of UAVs (1 = active, 0 = idle)

% Function to check if network remains bi-connected if UAV i is deleted
function rb = check_bi_connectivity(i, c_U, a)
    % Initialize rb to 1 (assuming the network is bi-connected initially)
    rb = 1;
    
    % Create the set of active UAVs excluding UAV i
    active_UAVs = find(a == 1); % Indices of active UAVs
    active_UAVs(active_UAVs == i) = []; % Remove the UAV being tested for deletion

    % Loop over each UAV in the active set to simulate its removal along with UAV i
    for index = 1:length(active_UAVs)
        r = active_UAVs(index);
        
        % Create a temporary connectivity matrix excluding UAVs i and r
        temp_c_U = c_U;
        temp_c_U(i, :) = 0; % Set row and column of UAV i to 0
        temp_c_U(:, i) = 0;
        temp_c_U(r, :) = 0; % Set row and column of UAV r to 0
        temp_c_U(:, r) = 0;

        % Get the remaining UAVs after excluding i and r
        remaining_UAVs = setdiff(active_UAVs, r);
        
        % Check connectivity among remaining UAVs using BFS
        connected = true;
        for j = 1:length(remaining_UAVs)
            for k = j+1:length(remaining_UAVs)
                % Perform BFS from UAV j to UAV k
                if ~bfs_connected(temp_c_U, remaining_UAVs(j), remaining_UAVs(k), remaining_UAVs)
                    connected = false;
                    % disp('Deleting the idle UAV would break the bi-connectivity of the network.');
                    break;
                end
            end
            if ~connected
                % disp('Deleting the idle UAV would break the bi-connectivity of the network.');
                break;
            end
        end
        
        % If any pair is not connected, set rb to 0 and exit
        if ~connected
            rb = 0;
            % disp('Deleting the idle UAV would break the bi-connectivity of the network.');
            return;
        end
    end
end

% Helper function to perform BFS and check connectivity between two nodes
function isConnected = bfs_connected(temp_c_U, start_node, end_node, valid_nodes)
    % Initialize BFS variables
    queue = [start_node];     % Queue to hold nodes for BFS traversal
    visited = false(size(temp_c_U, 1), 1); % Keep track of visited nodes
    visited(start_node) = true;

    % BFS loop
    while ~isempty(queue)
        current_node = queue(1); % Dequeue the front node
        queue(1) = [];           % Remove it from the queue
        
        % Check if we reached the target node
        if current_node == end_node
            isConnected = true;
            return;
        end
        
        % Find neighbors of the current node
        neighbors = find(temp_c_U(current_node, :) == 1);
        neighbors = intersect(neighbors, valid_nodes); % Restrict to valid nodes
        
        % Add unvisited neighbors to the queue
        for neighbor = neighbors
            if ~visited(neighbor)
                queue(end + 1) = neighbor; % Enqueue neighbor
                visited(neighbor) = true;
            end
        end
    end
    
    % If BFS completes without finding the end node, they are not connected
    isConnected = false;
end






%Algorithm 5 Delete Idle UAVs After Checking Constraints and Retrospect
% Parameters and Inputs from previous algorithms (assumed initialized)
% - U_r: Set of candidate UAVs for deletion
% - U_f: Set of idle UAVs that cannot be deleted due to constraints
% - c_U: Connectivity matrix among UAVs (N x N)
% - a: State vector of UAVs (1 = active, 0 = idle)
% - b_F, b_U: Association matrices between BSs/UAVs and UEs
% - C5, C6, C7: Boolean results for constraints (example placeholders here)
% - Fth: Resultant force threshold to stop UAVs
% - Rt: Coverage radius
% - Rc: Communication range threshold

% C5
function C5_or_not = C5(b_F, b_U, Connectivity_limit)
    totalSum_b_F = sum(b_F(:));
    totalSum_b_U = sum(b_U(:));
    total_b_FU = totalSum_b_F + totalSum_b_U;
    if  total_b_FU >= Connectivity_limit
        disp('Constraint C5 is satisfied.');
        C5_or_not = 1; % All elements are smaller than 1
    else
        disp('Constraint C5 is NOT satisfied.');
        C5_or_not = 0; % At least one element is not smaller than 1
    end
end

% C7
function C7_or_not = C7(c_F)
    % Calculate the sum of each column
    columnSums = sum(c_F, 1);
    % Count the number of columns with a sum greater than 0
    nonZeroColumns = (columnSums > 0);
    nonZeroColumns = double(nonZeroColumns);
    count = sum(nonZeroColumns);
    % Check if the constraint is satisfied
    if count >= 2
        disp('Constraint C7 is satisfied.');
        C7_or_not = 1;
    else
        disp('Constraint C7 is NOT satisfied.');
        C7_or_not = 0;
    end
end



% Function to delete idle UAVs after checking constraints and retrospect
function final_to_delete = delete_idle_uavs(U_r, U_f, c_U, c_F, a, b_F, b_U, Connectivity_limit)
    % Iterate over idle UAVs in U_r
    final_to_delete = U_r;
    for m = U_r
        c_F_modified = c_F;
        c_F_modified(:, m) = 0;

        b_U_modified = b_U;
        b_U_modified(m, :) = 0;

        % Execute Algorithm 4 to check constraint C6 (bi-connectivity)
        if check_bi_connectivity(m, c_U, a)
            % Check additional constraints C5 and C7
            if  C5(b_F, b_U_modified, Connectivity_limit) && C7(c_F_modified)

            
                % Delete the idle UAV m by setting its transmit power to 0
                disp('Deleting idle UAV ');
                disp(m);
                a(m) = 0; % Mark UAV m as deleted in state vector
                c_U(m, :) = 0; % Remove connections from UAV m in connectivity matrix
                c_U(:, m) = 0;
                c_F(:, m) = 0;
                                
                b_U(m, :) = 0; % ?? Remove associations between UAV m and all UEs

                % Update U_r and U_f
                % final_to_delete(final_to_delete == m) = []; % Remove UAV m from U_r
                U_f(U_f == m) = []; % Ensure UAV m is not in U_f
            else
                % If constraints C5 or C7 fail, add UAV m to U_f and keep active
                disp(['Keeping idle UAV ', num2str(m), ' due to constraint failure.']);
                U_f = unique([U_f, m]); % Add UAV m to U_f if it cannot be deleted
                a(m) = 1; % Ensure UAV m remains active
                final_to_delete(final_to_delete == m) = []; % Remove UAV m from U_r
            end
        else
            % If bi-connectivity check fails, add UAV m to U_f and keep active
            disp(['Bi-connectivity check failed for UAV ', num2str(m)]);
            U_f = unique([U_f, m]); % Add UAV m to U_f if it cannot be deleted
            a(m) = 1; % Ensure UAV m remains active
            final_to_delete(final_to_delete == m) = []; % Remove UAV m from U_r

        end
    end
end

% Example usage
tau=0.1;
Connectivity_limit = (1-tau)*M;
active = zeta_u ~= 0;
active = double(active);




% idle_UAV_index
U_r = find(active == 0)';
% result = check_bi_connectivity(idle_UAV, c_U, active);
U_f = [];        % Start with no idle UAVs that can't be deleted


% Execute the deletion algorithm
final_to_delete = delete_idle_uavs(U_r, U_f, c_U, c_F, active, b_F, b_U, Connectivity_limit);

% % Display final state of UAVs and sets
% disp('Final state vector (a):');
% disp(active);
% 
% disp('Final set of candidate UAVs for deletion (U_r):');
% disp(U_r);
% 
% disp('Final set of idle UAVs that cannot be deleted (U_f):');
% disp(U_f);

uav_positions(final_to_delete, :) = [];
plot_UE_network_connections(uav_positions, bs_positions, ue_positions, b_U, b_F, area_size)

plot_UAV_connections(uav_positions, bs_positions, ue_positions, c_U, c_F, area_size)


