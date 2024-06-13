from m_package.models.shallow_1D import lstm_1d_basic, conv_1d_basic, convlstm_1d_basic
from m_package.models.deep_1D import lstm_1d_deep, conv_1d_deep, convlstm_1d_deep
from m_package.models.models_2D import conv_2d_basic, conv_2d_deep
from m_package.models.depp_huddled_3D import conv3d_deep_huddled, convlstm_3d_deep_huddled
from m_package.models.shallow_huddled_3D import conv3d_basic_huddled, convlstm_3d_basic_huddled
from m_package.models.deep_3D import conv3d_deep, convlstm_3d_deep
from m_package.models.shallow_3D import conv3d_basic, convlstm_3d_basic


data_rep_mapping = {
    "_by_size": "3D",
    "_traj": "3D",
    "_huddled": "3D",
    "_img_fixation": "2D"
}


models_mapping = {
        ("1D", "basic", "lstm", ""): lstm_1d_basic,
        ("1D", "deep", "lstm", ""): lstm_1d_deep,
        ("1D", "basic", "convlstm", ""): convlstm_1d_basic,
        ("1D", "deep", "convlstm", ""): convlstm_1d_deep,
        ("1D", "basic", "conv", ""): conv_1d_basic,
        ("1D", "deep", "conv", ""): conv_1d_deep,
        ("2D", "basic", "", ""): conv_2d_basic,
        ("2D", "deep", "", ""): conv_2d_deep,
        ("3D", "basic", "conv", "_huddled"): conv3d_basic_huddled,
        ("3D", "basic", "conv", ""): conv3d_basic,
        ("3D", "basic", "convlstm", "_huddled"): convlstm_3d_basic_huddled,
        ("3D", "basic", "convlstm", ""): convlstm_3d_basic,
        ("3D", "deep", "conv", "_huddled"): conv3d_deep_huddled,
        ("3D", "deep", "conv", ""): conv3d_deep,
        ("3D", "deep", "convlstm", "_huddled"): convlstm_3d_deep_huddled,
        ("3D", "deep", "convlstm", ""): convlstm_3d_deep
}