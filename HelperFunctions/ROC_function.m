function conf_mat_per = ROC_function(reference, model, testing_set, set)
%clear, close all;
%reference = struct2table(reference);
rng(1)
if set == 1
     cv = cvpartition(reference.record_label,'KFold',5);
     testSet = reference(cv.test(1),:);
elseif set == 2
     cv = cvpartition(reference.record_label,'KFold',5);
     testSet = reference(cv.test(2),:);
elseif set == 3
     cv = cvpartition(reference.record_label,'KFold',5);
     testSet = reference(cv.test(3),:);
elseif set == 4
     cv = cvpartition(reference.record_label,'KFold',5);
     testSet = reference(cv.test(4),:);
elseif set == 5
     cv = cvpartition(reference.record_label,'KFold',5);
     testSet = reference(cv.test(5),:);
elseif set == 0
     testSet = reference;
end

%
predicted_class = predict(model, testing_set(:,1:28));
predicted_class = cell2table(num2cell(predicted_class));

%
result(:, 1) = testing_set(:, 29);
result(:, 2) = testing_set(:, 28);
result(:, 3) = predicted_class(:, 1);

%
for iwin = 1:length(testSet.record_name)
     %disp(iwin);
     if testSet.record_label(iwin) == 1
          class = 1;
     elseif testSet.record_label(iwin) == 2
          class = 2;
     end
     ab = 0;no = 0;
     for iwin2 = 1:length(result.Var1)
          if nnz(strcmp(testSet.record_name(iwin),result.Var1(iwin2))) == 1
               if result.Var3(iwin2) == 2
                    ab = ab + 1;
               elseif result.Var3(iwin2) == 1
                    no = no + 1;
               end
               result2.class(iwin, 1) = class;
               result2.ab(iwin, 1) = ab;
               result2.no(iwin, 1) = no;
          end
     end
end
result2 = struct2table(result2);

%
classflag = 0;
for iwin3 = 1:length(result2.class)
     % disp(iwin3);
     if result2.class(iwin3)~=0
          pro_ab = result2.ab(iwin3) / (result2.ab(iwin3) + result2.no(iwin3));
          result3.class(iwin3-classflag, 1) = result2.class(iwin3);
          if pro_ab > 0
               result3.p0(iwin3-classflag, 1) = 2;
          else
               result3.p0(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.05
               result3.p5(iwin3-classflag, 1) = 2;
          else
               result3.p5(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.1
               result3.p10(iwin3-classflag, 1) = 2;
          else
               result3.p10(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.15
               result3.p15(iwin3-classflag, 1) = 2;
          else
               result3.p15(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.2
               result3.p20(iwin3-classflag, 1) = 2;
          else
               result3.p20(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.25
               result3.p25(iwin3-classflag, 1) = 2;
          else
               result3.p25(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.3
               result3.p30(iwin3-classflag, 1) = 2;
          else
               result3.p30(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.35
               result3.p35(iwin3-classflag, 1) = 2;
          else
               result3.p35(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.4
               result3.p40(iwin3-classflag, 1) = 2;
          else
               result3.p40(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.45
               result3.p45(iwin3-classflag, 1) = 2;
          else
               result3.p45(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.5
               result3.p50(iwin3-classflag, 1) = 2;
          else
               result3.p50(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.55
               result3.p55(iwin3-classflag, 1) = 2;
          else
               result3.p55(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.6
               result3.p60(iwin3-classflag, 1) = 2;
          else
               result3.p60(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.65
               result3.p65(iwin3-classflag, 1) = 2;
          else
               result3.p65(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.7
               result3.p70(iwin3-classflag, 1) = 2;
          else
               result3.p70(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.75
               result3.p75(iwin3-classflag, 1) = 2;
          else
               result3.p75(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.8
               result3.p80(iwin3-classflag, 1) = 2;
          else
               result3.p80(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.85
               result3.p85(iwin3-classflag, 1) = 2;
          else
               result3.p85(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.9
               result3.p90(iwin3-classflag, 1) = 2;
          else
               result3.p90(iwin3-classflag, 1) = 1;
          end
          if pro_ab >= 0.95
               result3.p95(iwin3-classflag, 1) = 2;
          else
               result3.p95(iwin3-classflag, 1) = 1;
          end
          if pro_ab == 1
               result3.p100(iwin3-classflag, 1) = 2;
          else
               result3.p100(iwin3-classflag, 1) = 1;
          end
     else
          classflag = classflag + 1;
     end
end
result3 = struct2table(result3);

%
conf_mat0 = confusionmat(result3.class(:), result3.p0(:));
conf_mat_per0 = conf_mat0*100./sum(conf_mat0, 2);
conf_mat1 = confusionmat(result3.class(:), result3.p5(:));
conf_mat_per1 = conf_mat1*100./sum(conf_mat1, 2);
conf_mat2 = confusionmat(result3.class(:), result3.p10(:));
conf_mat_per2 = conf_mat2*100./sum(conf_mat2, 2);
conf_mat3 = confusionmat(result3.class(:), result3.p15(:));
conf_mat_per3 = conf_mat3*100./sum(conf_mat3, 2);
conf_mat4 = confusionmat(result3.class(:), result3.p20(:));
conf_mat_per4 = conf_mat4*100./sum(conf_mat4, 2);
conf_mat5 = confusionmat(result3.class(:), result3.p25(:));
conf_mat_per5 = conf_mat5*100./sum(conf_mat5, 2);
conf_mat6 = confusionmat(result3.class(:), result3.p30(:));
conf_mat_per6 = conf_mat6*100./sum(conf_mat6, 2);
conf_mat7 = confusionmat(result3.class(:), result3.p35(:));
conf_mat_per7 = conf_mat7*100./sum(conf_mat7, 2);
conf_mat8 = confusionmat(result3.class(:), result3.p40(:));
conf_mat_per8 = conf_mat8*100./sum(conf_mat8, 2);
conf_mat9 = confusionmat(result3.class(:), result3.p45(:));
conf_mat_per9 = conf_mat9*100./sum(conf_mat9, 2);
conf_mat10 = confusionmat(result3.class(:), result3.p50(:));
conf_mat_per10 = conf_mat10*100./sum(conf_mat10, 2);
conf_mat11 = confusionmat(result3.class(:), result3.p55(:));
conf_mat_per11 = conf_mat11*100./sum(conf_mat11, 2);
conf_mat12 = confusionmat(result3.class(:), result3.p60(:));
conf_mat_per12 = conf_mat12*100./sum(conf_mat12, 2);
conf_mat13 = confusionmat(result3.class(:), result3.p65(:));
conf_mat_per13 = conf_mat13*100./sum(conf_mat13, 2);
conf_mat14 = confusionmat(result3.class(:), result3.p70(:));
conf_mat_per14 = conf_mat14*100./sum(conf_mat14, 2);
conf_mat15 = confusionmat(result3.class(:), result3.p75(:));
conf_mat_per15 = conf_mat15*100./sum(conf_mat15, 2);
conf_mat16 = confusionmat(result3.class(:), result3.p80(:));
conf_mat_per16 = conf_mat16*100./sum(conf_mat16, 2);
conf_mat17 = confusionmat(result3.class(:), result3.p85(:));
conf_mat_per17 = conf_mat17*100./sum(conf_mat17, 2);
conf_mat18 = confusionmat(result3.class(:), result3.p90(:));
conf_mat_per18 = conf_mat18*100./sum(conf_mat18, 2);
conf_mat19 = confusionmat(result3.class(:), result3.p95(:));
conf_mat_per19 = conf_mat19*100./sum(conf_mat19, 2);
conf_mat20 = confusionmat(result3.class(:), result3.p100(:));
conf_mat_per20 = conf_mat20*100./sum(conf_mat20, 2);

TP0 = conf_mat_per0(4);FP0 = conf_mat_per0(3);FN0 = conf_mat_per0(2);TN0 = conf_mat_per0(1);
TP1 = conf_mat_per1(4);FP1 = conf_mat_per1(3);FN1 = conf_mat_per1(2);TN1 = conf_mat_per1(1);
TP2 = conf_mat_per2(4);FP2 = conf_mat_per2(3);FN2 = conf_mat_per2(2);TN2 = conf_mat_per2(1);
TP3 = conf_mat_per3(4);FP3 = conf_mat_per3(3);FN3 = conf_mat_per3(2);TN3 = conf_mat_per3(1);
TP4 = conf_mat_per4(4);FP4 = conf_mat_per4(3);FN4 = conf_mat_per4(2);TN4 = conf_mat_per4(1);
TP5 = conf_mat_per5(4);FP5 = conf_mat_per5(3);FN5 = conf_mat_per5(2);TN5 = conf_mat_per5(1);
TP6 = conf_mat_per6(4);FP6 = conf_mat_per6(3);FN6 = conf_mat_per6(2);TN6 = conf_mat_per6(1);
TP7 = conf_mat_per7(4);FP7 = conf_mat_per7(3);FN7 = conf_mat_per7(2);TN7 = conf_mat_per7(1);
TP8 = conf_mat_per8(4);FP8 = conf_mat_per8(3);FN8 = conf_mat_per8(2);TN8 = conf_mat_per8(1);
TP9 = conf_mat_per9(4);FP9 = conf_mat_per9(3);FN9 = conf_mat_per9(2);TN9 = conf_mat_per9(1);
TP10 = conf_mat_per10(4);FP10 = conf_mat_per10(3);FN10 = conf_mat_per10(2);TN10 = conf_mat_per10(1);
TP11 = conf_mat_per11(4);FP11 = conf_mat_per11(3);FN11 = conf_mat_per11(2);TN11 = conf_mat_per11(1);
TP12 = conf_mat_per12(4);FP12 = conf_mat_per12(3);FN12 = conf_mat_per12(2);TN12 = conf_mat_per12(1);
TP13 = conf_mat_per13(4);FP13 = conf_mat_per13(3);FN13 = conf_mat_per13(2);TN13 = conf_mat_per13(1);
TP14 = conf_mat_per14(4);FP14 = conf_mat_per14(3);FN14 = conf_mat_per14(2);TN14 = conf_mat_per14(1);
TP15 = conf_mat_per15(4);FP15 = conf_mat_per15(3);FN15 = conf_mat_per15(2);TN15 = conf_mat_per15(1);
TP16 = conf_mat_per16(4);FP16 = conf_mat_per16(3);FN16 = conf_mat_per16(2);TN16 = conf_mat_per16(1);
TP17 = conf_mat_per17(4);FP17 = conf_mat_per17(3);FN17 = conf_mat_per17(2);TN17 = conf_mat_per17(1);
TP18 = conf_mat_per18(4);FP18 = conf_mat_per18(3);FN18 = conf_mat_per18(2);TN18 = conf_mat_per18(1);
TP19 = conf_mat_per19(4);FP19 = conf_mat_per19(3);FN19 = conf_mat_per19(2);TN19 = conf_mat_per19(1);
TP20 = conf_mat_per20(4);FP20 = conf_mat_per20(3);FN20 = conf_mat_per20(2);TN20 = conf_mat_per20(1);

conf_mat_per.TP(1, 1) = TP0;conf_mat_per.FP(1, 1) = FP0;conf_mat_per.FN(1, 1) = FN0;conf_mat_per.TN(1, 1) = TN0;
conf_mat_per.TP(2, 1) = TP1;conf_mat_per.FP(2, 1) = FP1;conf_mat_per.FN(2, 1) = FN1;conf_mat_per.TN(2, 1) = TN1;
conf_mat_per.TP(3, 1) = TP2;conf_mat_per.FP(3, 1) = FP2;conf_mat_per.FN(3, 1) = FN2;conf_mat_per.TN(3, 1) = TN2;
conf_mat_per.TP(4, 1) = TP3;conf_mat_per.FP(4, 1) = FP3;conf_mat_per.FN(4, 1) = FN3;conf_mat_per.TN(4, 1) = TN3;
conf_mat_per.TP(5, 1) = TP4;conf_mat_per.FP(5, 1) = FP4;conf_mat_per.FN(5, 1) = FN4;conf_mat_per.TN(5, 1) = TN4;
conf_mat_per.TP(6, 1) = TP5;conf_mat_per.FP(6, 1) = FP5;conf_mat_per.FN(6, 1) = FN5;conf_mat_per.TN(6, 1) = TN5;
conf_mat_per.TP(7, 1) = TP6;conf_mat_per.FP(7, 1) = FP6;conf_mat_per.FN(7, 1) = FN6;conf_mat_per.TN(7, 1) = TN6;
conf_mat_per.TP(8, 1) = TP7;conf_mat_per.FP(8, 1) = FP7;conf_mat_per.FN(8, 1) = FN7;conf_mat_per.TN(8, 1) = TN7;
conf_mat_per.TP(9, 1) = TP8;conf_mat_per.FP(9, 1) = FP8;conf_mat_per.FN(9, 1) = FN8;conf_mat_per.TN(9, 1) = TN8;
conf_mat_per.TP(10, 1) = TP9;conf_mat_per.FP(10, 1) = FP9;conf_mat_per.FN(10, 1) = FN9;conf_mat_per.TN(10, 1) = TN9;
conf_mat_per.TP(11, 1) = TP10;conf_mat_per.FP(11, 1) = FP10;conf_mat_per.FN(11, 1) = FN10;conf_mat_per.TN(11, 1) = TN10;
conf_mat_per.TP(12, 1) = TP11;conf_mat_per.FP(12, 1) = FP11;conf_mat_per.FN(12, 1) = FN11;conf_mat_per.TN(12, 1) = TN11;
conf_mat_per.TP(13, 1) = TP12;conf_mat_per.FP(13, 1) = FP12;conf_mat_per.FN(13, 1) = FN12;conf_mat_per.TN(13, 1) = TN12;
conf_mat_per.TP(14, 1) = TP13;conf_mat_per.FP(14, 1) = FP13;conf_mat_per.FN(14, 1) = FN13;conf_mat_per.TN(14, 1) = TN13;
conf_mat_per.TP(15, 1) = TP14;conf_mat_per.FP(15, 1) = FP14;conf_mat_per.FN(15, 1) = FN14;conf_mat_per.TN(15, 1) = TN14;
conf_mat_per.TP(16, 1) = TP15;conf_mat_per.FP(16, 1) = FP15;conf_mat_per.FN(16, 1) = FN15;conf_mat_per.TN(16, 1) = TN15;
conf_mat_per.TP(17, 1) = TP16;conf_mat_per.FP(17, 1) = FP16;conf_mat_per.FN(17, 1) = FN16;conf_mat_per.TN(17, 1) = TN16;
conf_mat_per.TP(18, 1) = TP17;conf_mat_per.FP(18, 1) = FP17;conf_mat_per.FN(18, 1) = FN17;conf_mat_per.TN(18, 1) = TN17;
conf_mat_per.TP(19, 1) = TP18;conf_mat_per.FP(19, 1) = FP18;conf_mat_per.FN(19, 1) = FN18;conf_mat_per.TN(19, 1) = TN18;
conf_mat_per.TP(20, 1) = TP19;conf_mat_per.FP(20, 1) = FP19;conf_mat_per.FN(20, 1) = FN19;conf_mat_per.TN(20, 1) = TN19;
conf_mat_per.TP(21, 1) = TP20;conf_mat_per.FP(21, 1) = FP20;conf_mat_per.FN(21, 1) = FN20;conf_mat_per.TN(21, 1) = TN20;

end