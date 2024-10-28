NUM_SIGNALS = 10000;
mods = ["fm" "am"];
NUM_MODS = length(mods);

fs = 1000;
dt = 1/fs;
t = -0.5:dt:(0.5 - dt);

D = zeros(NUM_SIGNALS*NUM_MODS, length(t) + 1);
size(D)

for mod = 1:NUM_MODS
    for i = 1:NUM_SIGNALS
        w = 2*pi*(25*rand(1)+10);
        fc = (93*rand(1)+11);
    
        m = sin(w*t) + randn(size(t))/6;
        u = [modulate(m, fc, fs, mods(mod)) mod];
    
        D(NUM_SIGNALS*(mod-1) + i, :) = u;
    end
end

writematrix(D, "TrainData.csv")



