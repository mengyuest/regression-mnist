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

    lg_eta=0.3#2#0.001 #0.001
    lg_error_tol=20
    lg_T=12000#200000000#20000
    lg_reg=2#0#2
    lg_lamda=0.001#0.1
    lg_header="logistic"

    sm_eta=3.2#2.0#1.2
    sm_error_tol=20
    sm_T=20000#4000#8000
    sm_reg=0
    sm_lamda=0.00001
    sm_header="softmax"

    first2k=False
    cal_loss=False
    logistic=True
    softmax=False
    debug=True
    resdir="result/"
    mplotdir = "result/plots/"
    plotdir="result/plots/"

