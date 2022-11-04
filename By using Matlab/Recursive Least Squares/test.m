
load("simulink_workspace.mat");
sampletime = 1;
endtime =50;
numberofsamples = endtime *1/sampletime +1;
timeVector = (0:numberofsamples)*sampletime;

output = sim("simulink_model.slx",50);

error  = output.Signal_Estimation;
noise  = output.Noise;
noisy_signal = output.Noisy_Signal;
pure_signal = output.Pure_Signal;

figure
subplot(3,1,2);
plot(noise);
title("Noise");

subplot(3,1,1)
plot(noisy_signal);
hold on
plot(error);
title("Estimation");
legend("Corrupted Signal","Estimation");

subplot(3,1,3);
title("System Estimation")
plot(pure_signal);
hold on
plot(noisy_signal);
legend("Pure Signal","Corrupted Signal");