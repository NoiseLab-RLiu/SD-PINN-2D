minus_c2_rec = extractdata(parameters.UL1*parameters.SVR1);
alpha_rec = extractdata(parameters.UL2*parameters.SVR2);
figure
subplot(1,2,1)
imagesc(minus_c2_rec)
axis square
colorbar
xlabel('$\Delta x$','Interpreter','latex')
ylabel('$\Delta y$','Interpreter','latex')
ax=gca
ax.TickLabelInterpreter = 'latex'
subplot(1,2,2)
imagesc(alpha_rec)
axis square
colorbar
xlabel('$\Delta x$','Interpreter','latex')
ylabel('$\Delta y$','Interpreter','latex')
ax=gca
ax.TickLabelInterpreter = 'latex'