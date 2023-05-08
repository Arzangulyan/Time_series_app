def cwt():
    # #TEST
    # x = np.linspace(0, 1000,1000)
    # y = np.sin(2*np.pi*x/100) + np.cos(2*np.pi*x/300) + np.cos(2*np.pi*x/900)
    # linechart = st.line_chart(y)
    # coef, freqs = pywt.cwt(y, np.arange(1,129), 'mexh')
    # fig, ax = plt.subplots()
    # ax.imshow(coef, cmap = 'copper', aspect = 'auto')
    # st.pyplot(fig)
    # # sns.heatmap(coef, ax = ax, cmap = 'copper')
    # # st.write(fig)
    #
    # I = np.empty((len(freqs)))
    # for j in range(len(freqs)-1):
    #     for i in range(len(y)):
    #         I[j] += ((coef[j, i])**2 + (coef[j+1,i])**2)/2
    # # st.write(I)
    # I_s = pd.DataFrame({'I':I, 'Freqs': freqs})
    # st.write(I_s)
    # fig2, ax2 = plt.subplots()
    # ax2.plot(freqs, I)
    # ax2.set_aspect('auto')
    # st.pyplot(fig2)
    #
    # # Intergral_spectrum = st.line_chart(I_s)
    # #TEST

    # REAL

    coef, freqs = pywt.cwt(
        df_selected.loc[(start_point + m_a_step) : end_point, "Averaged"],
        np.arange(1, T_s_len / 4),
        mother_switcher.get(wavelet_select),
    )
    fig1, ax1 = plt.subplots()
    ax1.imshow(coef, cmap="copper", aspect="auto")
    # sns.heatmap(coef, ax = ax, cmap = 'copper')
    # st.write(fig)
    ax1.set_title("Power Spectrum", fontsize=20)
    ax1.set_ylabel("Период", fontsize=18)
    ax1.set_xlabel("Время", fontsize=18)
    ax1.invert_yaxis()
    st.pyplot(fig1)

    # I = np.empty((len(freqs)))
    # for j in range(len(freqs)-1):
    #     for i in range(len(df_selected.loc[(start_point+m_a_step):end_point, 'Averaged'])-1):
    #         I[j] += ((coef[j, i]) + (coef[j+1,i]))/2
    # # st.write(I)
    # Int_freq = pd.DataFrame({'I':I, 'Freqs': freqs})
    # st.write(Int_freq)
    # fig2, ax2 = plt.subplots()
    # ax2.plot(freqs, I)
    # ax2.set_aspect('auto')
    # plt.xscale("log")
    # st.pyplot(fig2)
    # REAL
