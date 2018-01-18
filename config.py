class cf:

    train_size=20000
    test_size=2000

    select_num1=2
    select_num2=3

    seed=1
    load_from_raw=True

    very_large_number=1000
    very_small_number=0.00000001

    train_ratio=0.9

    lg_eta=0.4#2#0.001 #0.001
    lg_error_tol=20
    lg_T=20000#200000000#20000
    lg_reg=2#0#2
    lg_lamda=0.001#0.1
    lg_header="logistic"

    sm_eta=1.0
    sm_error_tol=20
    sm_T=20000
    sm_reg=0
    sm_lamda=0.01
    sm_header="softmax"

    first2k=False
    cal_loss=False
    logistic=True
    softmax=False
    debug=True
    resdir="result/"
    plotdir="result/plots/"
