��=(      �sklearn.pipeline��Pipeline���)��}�(�steps�]�(�preprocessor��#sklearn.compose._column_transformer��ColumnTransformer���)��}�(�transformers�]�(�num��sklearn.preprocessing._data��StandardScaler���)��}�(�	with_mean���with_std���copy���_sklearn_version��1.6.0�ub]�(�flipper_length_mm��body_mass_g�e���cat��sklearn.preprocessing._encoders��OneHotEncoder���)��}�(�
categories��auto��sparse_output���dtype��numpy��float64����handle_unknown��error��drop�N�min_frequency�N�max_categories�N�feature_name_combiner��concat�hhub]�(�island��sex�e��e�	remainder�h-�sparse_threshold�G?�333333�n_jobs�N�transformer_weights�N�verbose���verbose_feature_names_out���force_int_remainder_cols���feature_names_in_��numpy._core.multiarray��_reconstruct���h(�ndarray���K ��Cb���R�(KK��h(�dtype����O8�����R�(K�|�NNNJ����J����K?t�b�]�(�island��bill_length_mm��bill_depth_mm��flipper_length_mm��body_mass_g��sex�et�b�n_features_in_�K�_columns�]�(hh2e�_transformer_to_input_indices�}�(h]�(KKeh]�(K Keh6]�(KKeu�
_remainder�h6h-h�_RemainderColsList���)��}�(�data�]�(KKe�future_dtype��str��warning_was_emitted���warning_enabled��ub���sparse_output_���transformers_�]�(hh)��}�(h�h�h�h=h@hBK ��hD��R�(KK��hL�]�(hShTet�bhWK�n_samples_seen_�h>�scalar���hI�i8�����R�(K�<�NNNJ����J����K t�bC
      ���R��mean_�h@hBK ��hD��R�(KK��hI�f8�����R�(Kh|NNNJ����J����K t�b�C�5�� i@�L��a�@�t�b�var_�h@hBK ��hD��R�(KK��h��C_�̃u�g@���U�#A�t�b�scale_�h@hBK ��hD��R�(KK��h��C#�Gy��+@��?c���@�t�bhhubh��hh!)��}�(h$h%h&�h'h*h+h,h-Nh.Nh/Nh0h1�_infrequent_enabled��hWKh=h@hBK ��hD��R�(KK��hL�]�(hPhUet�b�categories_�]�(h@hBK ��hD��R�(KK��hL�]�(�Biscoe��Dream��	Torgersen�et�bh@hBK ��hD��R�(KK��hL�]�(�Female��Male�et�be�_drop_idx_after_grouping�N�	drop_idx_�N�_n_features_outs�]�(KKehhubh2��h6h-ha)��}�(hd]�(KKehfhghh�hi�ub��e�output_indices_�}�(h�builtins��slice���K KN��R�hh�KKN��R�h6h�K K N��R�uhhub���
classifier��sklearn.tree._classes��DecisionTreeClassifier���)��}�(�	criterion��gini��splitter��best��	max_depth�N�min_samples_split�K�min_samples_leaf�K�min_weight_fraction_leaf�G        �max_features�N�max_leaf_nodes�N�random_state�K�min_impurity_decrease�G        �class_weight�N�	ccp_alpha�G        �monotonic_cst�NhWK�
n_outputs_�K�classes_�h@hBK ��hD��R�(KK��hL�]�(�Adelie��	Chinstrap��Gentoo�et�b�
n_classes_�hxh{C       ���R��max_features_�K�tree_��sklearn.tree._tree��Tree���Kh@hBK ��hD��R�(KK��h{�C       �t�bK��R�}�(h�K�
node_count�KS�nodes�h@hBK ��hD��R�(KKS��hI�V64�����R�(KhMN(�
left_child��right_child��feature��	threshold��impurity��n_node_samples��weighted_n_node_samples��missing_go_to_left�t�}�(j  h{K ��j  h{K��j	  h{K��j
  h�K��j  h�K ��j  h{K(��j  h�K0��j  hI�u1�����R�(KhMNNNJ����J����K t�bK8��uK@KKt�b�B�         L                  �[P�?�<��h�?
           �p@                                  �?z����y�?�            �d@                                  �B�? �й���?I            @R@       ������������������������       �        G            �Q@                                   �?      �?              @        ������������������������       �                     �?        ������������������������       �                     �?               -                  �3����MΖ��?]            @W@        	       ,                 ����q�q�?-            �F@       
                        ����և���X�?#            �A@                                 u_��LQ�1	�?             7@                                 `P���      �?              @                                ���      �?              @        ������������������������       �                     �?        ������������������������       �                     �?        ������������������������       �                     @                                ��$����S���?             .@        ������������������������       �                      @                                  W返n_Y�K�?             *@                               p��      �?
             $@                                0�F��q�q�?             @        ������������������������       �                     @                                0f��q�q�?             @       ������������������������       �                      @        ������������������������       �                     �?                                 ���      �?             @        ������������������������       �                      @        ������������������������       �      �?              @        ������������������������       �                     @                                ����q�q�?             (@        ������������������������       �                      @                +                    �?      �?
             $@       !       "                   V���X�<ݚ�?	             "@        ������������������������       �                     �?        #       $                   u_�      �?              @        ������������������������       �                     @        %       *                   W迸��Q��?             @       &       )                  ���      �?             @        '       (                 ����      �?              @        ������������������������       �                     �?        ������������������������       �                     �?        ������������������������       �                      @        ������������������������       �                     �?        ������������������������       �                     �?        ������������������������       �        
             $@        .       A                 @)�ҿr�qG�?0             H@       /       :                  p��ٿ@�0�!��?"             A@        0       3                  Պ��q�q�?             .@        1       2                 �x��      �?             @        ������������������������       �                     �?        ������������������������       �                     @        4       9                 P��ٿ"pc�
�?             &@       5       6                 @��ףp=
�?
             $@        ������������������������       �                     @        7       8                 P1o�z�G�z�?             @        ������������������������       �                     �?        ������������������������       �                     @        ������������������������       �                     �?        ;       <                 ��f��?�}�+r��?             3@       ������������������������       �                     .@        =       @                 ����      �?             @        >       ?                 ����      �?              @        ������������������������       �                     �?        ������������������������       �                     �?        ������������������������       �                      @        B       E                 ��x��      �?             ,@        C       D                    �?r�q��?             @        ������������������������       �                     �?        ������������������������       �                     @        F       G                 ��H˿      �?              @        ������������������������       �                     �?        H       I                  �(��?؇���X�?             @       ������������������������       �                     @        J       K                 P��?      �?              @        ������������������������       �                     �?        ������������������������       �                     �?        M       R                    �?���QI�?d             Y@        N       O                  �p�?�q�q�?             @        ������������������������       �                     �?        P       Q                    �?z�G�z�?             @        ������������������������       �                     �?        ������������������������       �                     @        ������������������������       �        ^            �W@        �t�b�values�h@hBK ��hD��R�(KKSKK��h��B�  h	&�?	&��?�m۶m��?���C.+�?]V��F�?��k��x?����Ǐ�?        ����?      �?                      �?              �?                      �?      �?                �]v�e��?4�DM4�?        UUUUUU�?UUUUUU�?        �$I�$I�?۶m۶m�?        Nozӛ��?d!Y�B�?              �?      �?              �?      �?              �?                              �?              �?                �?�������?                      �?        ;�;��?ى�؉��?              �?      �?        UUUUUU�?UUUUUU�?                      �?        UUUUUU�?UUUUUU�?              �?                              �?              �?      �?              �?                      �?      �?              �?                �������?�������?                      �?              �?      �?        �q�q�?r�q��?              �?                      �?      �?                      �?        333333�?�������?              �?      �?              �?      �?                      �?              �?                      �?                              �?              �?                      �?                UUUUUU�?UUUUUU�?        �������?ZZZZZZ�?        UUUUUU�?UUUUUU�?              �?      �?                      �?              �?                F]t�E�?/�袋.�?        �������?�������?                      �?        �������?�������?              �?                              �?              �?                (�����?�5��P�?                      �?              �?      �?              �?      �?                      �?              �?                              �?              �?      �?        �������?UUUUUU�?                      �?              �?                      �?      �?              �?                �$I�$I�?۶m۶m�?                      �?              �?      �?              �?                              �?        {�G�z�?{�G�z�?�G�z�?UUUUUU�?UUUUUU�?              �?                �������?�������?              �?                              �?                              �?�t�bubhhub��e�transform_input�N�memory�Nh:�hhub.