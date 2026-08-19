%% roll_out Live Script Demo – Advanced Control for Robotics
% This live-style script demonstrates the unified `roll_out(model, ctrl, x0, sim, opts)` API
% across common tutorial scenarios:
%   1) CT vs DT (Euler/ZOH/RK4) for a first-order system
%   2) ZOH vs Euler vs RK4 for a double integrator (position & velocity split)
%   3) 2D vehicle (unicycle3) trajectory
%   4) 7-DOF manipulator trajectory (requires Robotics System Toolbox)
%   5) Joint-space vs Cartesian-space control (manipulator; requires RST)
%   6) Measurement-feedback and a simple observer
%   7) Time-varying dynamics via opts.timevarying
%
% HOW TO USE:
%   • Run as a normal script, or open in Live Editor and "Convert to Live Script".
%   • Ensure these functions are on your MATLAB path:
%       - roll_out.m (new API), resolve_dynamics.m, step.m (with Euler/ZOH/RK4/Direct)
%       - your f(x,u,name).m and getAB(name,...).m for named models
%
% Tip: In Live Editor, you can add sliders for dt/K/Tmax to make it interactive.

%% Setup
clear; clc; rng(0);
fprintf('[roll_out demo] MATLAB %s, %s\n', version, datestr(now));

% Color scheme for integrators
colCT = [0 0 0];      % black
colZ  = [0.85 0.33 0.10];   % orange/red-ish
colE  = [0 0.45 0.74];      % blue-ish
colR  = [0.47 0.67 0.19];   % green-ish

%% 1) CT vs DT (Euler/ZOH/RK4): single integrator
name = "single_integrator";
x0   = 0;
Tmax = 5;
ctrl = @(x,t) 5 - x;

% CT simulation
simCT = struct('type','CT','T',Tmax,'integrator','ode45');
[t_ct, x_ct, u_ct] = roll_out(name, ctrl, x0, simCT);

% DT simulations
dt = 0.1;  K = round(Tmax/dt);
simE = struct('type','DT','K',K,'dt',dt,'integrator','Euler');
simZ = struct('type','DT','K',K,'dt',dt,'integrator','ZOH');
simR = struct('type','DT','K',K,'dt',dt,'integrator','RK4');
[t_e, x_e, u_e] = roll_out(name, ctrl, x0, simE);
[t_z, x_z, u_z] = roll_out(name, ctrl, x0, simZ);
[t_r, x_r, u_r] = roll_out(name, ctrl, x0, simR);

figure('Name','1) CT vs DT – single integrator'); clf;
subplot(2,1,1); hold on; box on;
plot(t_ct, x_ct, 'Color', colCT, 'LineWidth', 1.25);
plot(t_z,  x_z,  'Color', colZ);
plot(t_e,  x_e,  'Color', colE);
plot(t_r,  x_r,  'Color', colR);
legend('CT (ode45)', sprintf('ZOH dt=%.3g',dt), sprintf('Euler dt=%.3g',dt), sprintf('RK4 dt=%.3g',dt), 'Location','best');
title('State trajectory (x)'); xlabel('t (s)'); ylabel('x');

subplot(2,1,2); hold on; box on;
plot(t_ct(1:end-1), u_ct, 'Color', colCT, 'LineWidth', 1.25);
plot(t_z(1:end-1),  u_z,  'Color', colZ);
plot(t_e(1:end-1),  u_e,  'Color', colE);
plot(t_r(1:end-1),  u_r,  'Color', colR);
legend('CT (ode45)', 'ZOH', 'Euler', 'RK4', 'Location','best');
title('Control signal u(t_k)'); xlabel('t (s)'); ylabel('u');

%% 2) ZOH vs Euler vs RK4: double integrator (position & velocity)
name = "double_integrator";
x0   = [0;0];
Tmax = 5;
ctrl = @(x,t) 1;   % try sin(t) or 5 - x(1) - 2*x(2)

% CT simulation
simCT = struct('type','CT','T',Tmax,'integrator','ode45');
[t_ct, x_ct, u_ct] = roll_out(name, ctrl, x0, simCT);

% DT simulations
dt = 0.5;  K = round(Tmax/dt);
simE = struct('type','DT','K',K,'dt',dt,'integrator','Euler');
simZ = struct('type','DT','K',K,'dt',dt,'integrator','ZOH');
simR = struct('type','DT','K',K,'dt',dt,'integrator','RK4');
[t_e, x_e, u_e] = roll_out(name, ctrl, x0, simE);
[t_z, x_z, u_z] = roll_out(name, ctrl, x0, simZ);
[t_r, x_r, u_r] = roll_out(name, ctrl, x0, simR);

figure('Name','2) ZOH vs Euler vs RK4 – double integrator'); clf;

% Position
subplot(3,1,1); hold on; box on;
plot(t_ct, x_ct(1,:), 'Color', colCT, 'LineWidth', 1.25);
plot(t_z,  x_z(1,:),  'Color', colZ);
plot(t_e,  x_e(1,:),  'Color', colE);
plot(t_r,  x_r(1,:),  'Color', colR);
legend('CT (ode45)', sprintf('ZOH dt=%.3g',dt), sprintf('Euler dt=%.3g',dt), sprintf('RK4 dt=%.3g',dt),'Location','best');
title('Position x_1(t)'); xlabel('t (s)'); ylabel('x_1');

% Velocity
subplot(3,1,2); hold on; box on;
plot(t_ct, x_ct(2,:), 'Color', colCT, 'LineWidth', 1.25);
plot(t_z,  x_z(2,:),  'Color', colZ);
plot(t_e,  x_e(2,:),  'Color', colE);
plot(t_r,  x_r(2,:),  'Color', colR);
legend('CT (ode45)', 'ZOH', 'Euler', 'RK4','Location','best');
title('Velocity x_2(t)'); xlabel('t (s)'); ylabel('x_2');

% Control
subplot(3,1,3); hold on; box on;
plot(t_ct(1:end-1), u_ct, 'Color', colCT, 'LineWidth', 1.25);
plot(t_z(1:end-1),  u_z,  'Color', colZ);
plot(t_e(1:end-1),  u_e,  'Color', colE);
plot(t_r(1:end-1),  u_r,  'Color', colR);
legend('CT (ode45)', 'ZOH', 'Euler', 'RK4','Location','best');
title('Control signal'); xlabel('t (s)'); ylabel('u');

%% 3) 2D vehicle trajectory (unicycle3)
name = "unicycle3";
x0   = [0;0;0];
Tmax = 5;
goal = [5;5]; kp = 1; ktheta = 1;
ctrl = @(x,t) [ kp*dot(goal - x(1:2), [cos(x(3)), sin(x(3))]); ...
                ktheta*(atan2(goal(2)-x(2), goal(1)-x(1)) - x(3)) ];

dt = 0.1;  K = round(Tmax/dt);
sim = struct('type','DT','K',K,'dt',dt,'integrator','Euler');
[t3, x3, u3] = roll_out(name, ctrl, x0, sim);

figure('Name','3) Unicycle trajectory'); clf; hold on; axis equal; box on;
plot(x3(1,:), x3(2,:), 'k', 'LineWidth', 1.2);
scatter(x0(1), x0(2), 36, 'b', 'filled'); text(x0(1), x0(2), ' start','Color','b');
scatter(goal(1), goal(2), 36, 'r', 'filled'); text(goal(1), goal(2), ' goal','Color','r');
title('Unicycle3 trajectory'); xlabel('x'); ylabel('y'); grid on;

%% 4) Manipulator trajectory (requires Robotics System Toolbox)
try
    global robot;
    robot = loadrobot('kinovaGen3','DataFormat','row','Gravity',[0 0 -9.81]);
    q0    = homeConfiguration(robot);
    name  = "single_integrator";  % joint-space single-integrator
    Tmax  = 5;
    ctrl  = @(q,t) 0.3*randn(7,1);   % small random joint velocity command

    dt = 0.05; K = round(Tmax/dt);
    sim = struct('type','DT','K',K,'dt',dt,'integrator','Euler');
    [t4, qhist, uhist] = roll_out(name, ctrl, q0', sim);

    % Compute EE path
    endEffector = "EndEffector_Link";
    p = zeros(3, numel(t4));
    q = q0;
    for k=1:numel(t4)
        q = qhist(:,k)';
        T = getTransform(robot, q, endEffector);
        p(:,k) = tform2trvec(T).';
    end

    figure('Name','4) Manipulator EE path'); clf; hold on; box on;
    plot3(p(1,:), p(2,:), p(3,:), 'k', 'LineWidth', 1.2);
    scatter3(p(1,1), p(2,1), p(3,1), 36, 'b', 'filled'); text(p(1,1), p(2,1), p(3,1), ' start','Color','b');
    title('End-effector path (Kinova Gen3)'); xlabel('x'); ylabel('y'); zlabel('z'); grid on; axis vis3d;

catch ME
    warning('[Manipulator demos skipped] %s', ME.message);
end

%% 5) Joint-space vs Cartesian-space control (requires Robotics System Toolbox)
try
    global robot;
    robot = loadrobot('kinovaGen3','DataFormat','row','Gravity',[0 0 -9.81]);

    goal = [0.4, 0, 0.6];
    q0   = homeConfiguration(robot);
    ee   = "EndEffector_Link";
    T0   = getTransform(robot, q0, ee);
    ik   = inverseKinematics("RigidBodyTree", robot);
    qT   = ik(ee, trvec2tform(goal)*axang2tform([0 1 0 pi]), [1 1 1 1 1 1], q0); qT = mod(qT, 2*pi);

    Tmax = 5; dt = 0.1; K = round(Tmax/dt);
    name = "single_integrator";

    % 5.1 Joint-space control u = K*(qT - q)
    ctrlJ = @(q,t) 0.7*(qT' - q);
    sim   = struct('type','DT','K',K,'dt',dt,'integrator','Euler');
    [t5j, qhistJ] = roll_out(name, ctrlJ, q0', sim);
    pJ = zeros(3, numel(t5j));
    for k=1:numel(t5j)
        pJ(:,k) = tform2trvec(getTransform(robot, qhistJ(:,k)', ee)).';
    end

    % 5.2 Cartesian-space control via Jacobian pseudo-inverse
    name = "arm_first_order";  % uses your name-based Jacobian model
    ctrlC = @(x, t) 0.5 .* pinv( geometricJacobian(robot, ...
        ik(ee, trvec2tform(x'), [1 1 1 1 1 1], homeConfiguration(robot)), ee) ) * [goal' - x; zeros(3,1)];
    [t5c, xhistC] = roll_out(name, ctrlC, tform2trvec(T0)', sim);

    figure('Name','5) Joint vs Cartesian control'); clf; hold on; box on;
    plot3(pJ(1,:), pJ(2,:), pJ(3,:), 'Color', colE, 'LineWidth', 1.4);
    plot3(xhistC(1,:), xhistC(2,:), xhistC(3,:), 'Color', colR, 'LineWidth', 1.4);
    scatter3(goal(1), goal(2), goal(3), 48, 'r', 'filled'); text(goal(1),goal(2),goal(3),' goal','Color','r');
    legend('Joint-space control','Cartesian-space control','goal','Location','best');
    title('End-effector paths'); xlabel('x'); ylabel('y'); zlabel('z'); grid on; axis vis3d;

catch ME
    warning('[Joint vs Cartesian demos skipped] %s', ME.message);
end

%% 6) Measurement-feedback & simple observer (double integrator)
name = "double_integrator";
x0   = [0.5; 0];
dt   = 0.01; K = 600;
C    = [1 0]; vstd = 0.03;
opts  = struct;
opts.sensor = @(x,~,~) C*x + vstd*randn;   % y = Cx + noise

% Observer (toy Luenberger-like)
A = [1 dt; 0 1]; B = [0.5*dt^2; dt]; L = [0.6; 6];
opts.observer = struct('xhat0', [0;0], ...
    'update', @(xhat,u,y,t) A*xhat + B*u + L*(y - C*xhat) );

ctrl = struct('mode','estimate', 'pi', @(xhat,t) -[2 1]*xhat);
sim  = struct('type','DT','K',K,'dt',dt,'integrator','RK4');
opts.log_fields = {'y','xhat'};

[t6, x6, u6, log6] = roll_out(name, ctrl, x0, sim, opts);

figure('Name','6) Measurement & Observer'); clf;
subplot(3,1,1); hold on; box on;
plot(t6, x6(1,:), 'k','LineWidth',1.2);
plot(t6, log6.xhat(1,:), 'Color', colR);
legend('x_1 (true)','xhat_1 (estimate)','Location','best');
title('Position'); xlabel('t (s)');

subplot(3,1,2); hold on; box on;
plot(t6, x6(2,:), 'k','LineWidth',1.2);
plot(t6, log6.xhat(2,:), 'Color', colR);
legend('x_2 (true)','xhat_2 (estimate)','Location','best');
title('Velocity'); xlabel('t (s)');

subplot(3,1,3); hold on; box on;
stairs(t6(1:end-1), u6, 'Color', colE);
if isfield(log6,'y')
    yyaxis right; plot(t6(1:end-1), log6.y, 'Color', colZ);
    ylabel('measurement y');
    yyaxis left;
end
legend('u','y','Location','best');
title('Control & Measurement'); xlabel('t (s)'); ylabel('u');

%% 7) Time-varying dynamics via opts.timevarying (single integrator with drift after 2s)
% We create a numeric CT model whose RHS is replaced per step by timevarying hook.
mdl = struct('f', @(x,u) u);  % start identical to single integrator
x0  = 0; dt = 0.02; Tmax = 6; K = round(Tmax/dt);
ctrl = @(x,t) 1;              % nominal command

% After t >= 2s, add a negative drift term -a*x with a=0.8 (e.g., a spring)
opts = struct;
opts.timevarying = @(dyn,k,t) setfield(dyn,'f_ct', @(x,u) (u - (t>=2)*0.8*x)); %#ok<SFLD>
sim  = struct('type','DT','K',K,'dt',dt,'integrator','RK4');
[t7, x7, u7] = roll_out(mdl, ctrl, x0, sim, opts);

figure('Name','7) Time-varying dynamics'); clf; hold on; box on;
plot(t7, x7, 'Color', colR, 'LineWidth', 1.2);
xline(2,'--','t=2s (drift on)');
title('Single integrator with time-varying drift'); xlabel('t (s)'); ylabel('x');

% End of demo
