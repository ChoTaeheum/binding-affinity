import numpy as np
from matplotlib import pyplot as plt


desc_chemception_tr = np.load('results/desc+chemception_tr_losses.npy')
desc_chemception_va = np.load('results/desc+chemception_va_losses.npy')
desc_ecfp_tr = np.load('results/desc+ecfp_tr_losses.npy')
desc_ecfp_va = np.load('results/desc+ecfp_va_losses.npy')
desc_gcn_tr = np.load('results/desc+gcn_tr_losses.npy')
desc_gcn_va = np.load('results/desc+gcn_va_losses.npy')
desc_gcnlstm_tr = np.load('results/desc+gcnlstm_tr_losses.npy')
desc_gcnlstm_va = np.load('results/desc+gcnlstm_va_losses.npy')
desc_vanilacnn_tr = np.load('results/desc+vanilacnn_tr_losses.npy')[:300]
desc_vanilacnn_va = np.load('results/desc+vanilacnn_va_losses.npy')[:300]
lstm_ecfp_tr = np.load('results/lstm+ecfp_tr_losses.npy')
lstm_ecfp_va = np.load('results/lstm+ecfp_va_losses.npy')
lstm_gcn_tr = np.load('results/lstm+gcn_tr_losses.npy')
lstm_gcn_va = np.load('results/lstm+gcn_va_losses.npy')
lstm_chemception_tr = np.load('results/lstm+chemception_tr_losses.npy')
lstm_chemception_va = np.load('results/lstm+chemception_va_losses.npy')


modified_v6_tr = np.load('results/ModifiedDeepDTA_v6_tr_losses.npy')
modified_v6_va = np.load('results/ModifiedDeepDTA_v6_va_losses.npy')


alg_list = ['desc_chemception_tr', 'desc_chemception_va', 'desc_ecfp_tr', 'desc_ecfp_va', 'desc_gcn_tr', 'desc_gcn_va',
            'desc_gcnlstm_tr', 'desc_gcnlstm_va', 'desc_vanilacnn_tr', 'desc_vanilacnn_va', 'lstm_ecfp_tr', 'lstm_ecfp_va',
            'lstm_gcn_tr', 'lstm_gcn_va', 'lstm_chemception_tr', 'lstm_chemception_va']

scale_list = [0.7886, 0.8895, 0.7181, 0.9506, 0.6990, 0.6201, 0.6990, 0.6201, 0.7886, 0.8895, 0.7874, 0.9287, 0.7878, 0.9149, 0.7882, 0.9048]

# saturation
for i, algorithm in enumerate(alg_list):
    exec("alg = {}".format(algorithm))
    alg = alg / scale_list[i]
    alg[alg>3] = 3
    exec("{} = alg".format(algorithm))
    

epochs = np.arange(300)
# =============================================================================
plt.plot(epochs, desc_chemception_tr, desc_chemception_va)
plt.axis([-10, 310, -0.2, 3.2])
plt.minorticks_on()
plt.title('desc&chemception')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.grid(True)
tr_min = np.min(np.round(desc_chemception_tr, 4))
va_min = np.min(np.round(desc_chemception_va, 4))
# =============================================================================

# =============================================================================
plt.plot(epochs, desc_ecfp_tr, desc_ecfp_va)
plt.axis([-10, 310, -0.2, 3.2])
plt.minorticks_on()
plt.title('desc&ecfp')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.grid(True)
tr_min = np.min(np.round(desc_ecfp_tr, 4))
va_min = np.min(np.round(desc_ecfp_va, 4))
# =============================================================================

# =============================================================================
plt.plot(epochs, desc_gcn_tr, desc_gcn_va)
plt.axis([-10, 310, -0.2, 3.2])
plt.minorticks_on()
plt.title('desc&gcn')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.grid(True)
tr_min = np.min(np.round(desc_gcn_tr, 4))
va_min = np.min(np.round(desc_gcn_va, 4))
# =============================================================================

# =============================================================================
plt.plot(epochs, desc_gcnlstm_tr, desc_gcnlstm_va)
plt.minorticks_on()
plt.title('desc&gcnlstm')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.grid(True)
tr_min = np.min(np.round(desc_gcnlstm_tr, 4))
va_min = np.min(np.round(desc_gcnlstm_va, 4))
# =============================================================================

# =============================================================================
aa = plt.plot(epochs, desc_vanilacnn_tr, desc_vanilacnn_va)
plt.axis([-10, 310, -0.2, 3.2])
plt.minorticks_on()
plt.title('desc&vanilacnn')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.grid(True)
tr_min = np.min(np.round(desc_vanilacnn_tr, 4))
va_min = np.min(np.round(desc_vanilacnn_va, 4))
# =============================================================================

# =============================================================================
plt.plot(epochs, lstm_ecfp_tr, lstm_ecfp_va)
plt.axis([-10, 310, -0.2, 3.2])
plt.minorticks_on()
plt.title('lstm&ecfp')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.grid(True)
tr_min = np.min(np.round(lstm_ecfp_tr, 4))
va_min = np.min(np.round(lstm_ecfp_va, 4))
# =============================================================================

# =============================================================================
plt.plot(epochs, lstm_gcn_tr, lstm_gcn_va)
plt.axis([-10, 310, -0.2, 3.2])
plt.minorticks_on()
plt.title('lstm&gcn')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.grid(True)
tr_min = np.min(np.round(lstm_gcn_tr, 4))
va_min = np.min(np.round(lstm_gcn_va, 4))
# =============================================================================

# =============================================================================
plt.plot(epochs, lstm_chemception_tr, lstm_chemception_va)
plt.axis([-10, 310, -0.2, 3.2])
plt.minorticks_on()
plt.title('lstm&chemception')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.grid(True)
tr_min = np.min(np.round(lstm_chemception_tr, 4))
va_min = np.min(np.round(lstm_chemception_va, 4))
# =============================================================================

# =============================================================================
plt.plot(epochs, modified_v6_tr, modified_v6_va)
plt.axis([-10, 111, -0.2, 3.2])
plt.minorticks_on()
plt.title('Modified_DeepDTA_6')
plt.xlabel('epochs')
plt.ylabel('losses')
plt.grid(True)
tr_min = np.min(np.round(modified_v6_tr, 4))
va_min = np.min(np.round(modified_v6_va, 4))
# =============================================================================
