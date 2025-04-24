#include <trajectory.hpp>
Trajectory t(0, std::vector<SamplePoint>{
                    SamplePoint(3, 15.5, 1),     SamplePoint(5, 15.5, 2),    SamplePoint(7, 15.5, 3),
                    SamplePoint(8.5, 15.5, 4),  SamplePoint(9.5, 15.5, 5), SamplePoint(10, 15.5, 6),
                    SamplePoint(11.5, 15.5, 7), SamplePoint(12, 14, 8),    SamplePoint(12, 12, 9),
                    SamplePoint(12, 11, 10),     SamplePoint(12, 10, 11),    SamplePoint(12, 8, 12),
                    SamplePoint(12, 5.5, 13),    SamplePoint(13, 4, 14),     SamplePoint(14, 3, 15),
                    SamplePoint(14, 2, 16),      SamplePoint(16, 2, 17),     SamplePoint(18.5, 2, 18),
                    SamplePoint(20.5, 2, 19),    SamplePoint(21.5, 2, 20)});

Trajectory t_copy(10, t.points);

Trajectory t1(1,
              std::vector<SamplePoint>{SamplePoint(2, 2.5, 6), SamplePoint(1.5, 3, 7), SamplePoint(1.5, 4, 8),
                                       SamplePoint(1.5, 5.5, 9), SamplePoint(1.5, 7, 10), SamplePoint(1.5, 8.5, 11),
                                       SamplePoint(1.5, 9.5, 12), SamplePoint(1.5, 10.5, 13), SamplePoint(1.5, 12, 14),
                                       SamplePoint(1.5, 13, 15), SamplePoint(2, 14, 16), SamplePoint(3, 14.55, 17),
                                       SamplePoint(5, 14.55, 18), SamplePoint(7, 14.55, 19)});

Trajectory t2(2, std::vector<SamplePoint>{SamplePoint(5, 16, 2), SamplePoint(7.5, 16, 3), SamplePoint(8.5, 16, 4),
                                          SamplePoint(9.5, 16, 5), SamplePoint(12, 15.5, 6), SamplePoint(12.5, 14.5, 7),
                                          SamplePoint(12.5, 13.5, 8), SamplePoint(12.5, 12, 9),
                                          SamplePoint(14, 11.5, 10), SamplePoint(15, 11, 11), SamplePoint(16.5, 11, 12),
                                          SamplePoint(17.5, 11, 13), SamplePoint(18.5, 11, 14),
                                          SamplePoint(19.5, 11.5, 15), SamplePoint(19.5, 12, 16),
                                          SamplePoint(19.5, 13.5, 17), SamplePoint(19.5, 15, 18)});

Trajectory t3(3, std::vector<SamplePoint>{SamplePoint(5.5, 14, 13), SamplePoint(7, 14, 14), SamplePoint(8, 14.5, 15),
                                          SamplePoint(10, 14.5, 16), SamplePoint(11, 14, 17), SamplePoint(11.5, 12, 18),
                                          SamplePoint(9.5, 12, 19), SamplePoint(8, 12, 20), SamplePoint(6, 12, 21),
                                          SamplePoint(4.5, 12, 22), SamplePoint(3.5, 12, 23), SamplePoint(2.5, 11, 24),
                                          SamplePoint(2, 10.5, 25), SamplePoint(2, 9, 26), SamplePoint(2, 8, 27)});

Trajectory t4(4,
              std::vector<SamplePoint>{SamplePoint(5.58, 11, 6), SamplePoint(6.58, 11, 7), SamplePoint(7.57, 11, 8),
                                       SamplePoint(8.58, 11, 9), SamplePoint(9.55, 11, 10), SamplePoint(11.5, 11, 11),
                                       SamplePoint(11.5, 10, 12), SamplePoint(11.5, 8.5, 13), SamplePoint(11.5, 8, 14),
                                       SamplePoint(11.5, 6, 15), SamplePoint(11.5, 5.11, 16),
                                       SamplePoint(12.5, 4.1, 17), SamplePoint(13, 3.5, 18), SamplePoint(13.5, 3, 19),
                                       SamplePoint(13.5, 2, 20), SamplePoint(13.5, 1, 21), SamplePoint(12.5, 1, 22),
                                       SamplePoint(10.5, 1, 23), SamplePoint(8.5, 1, 24)});

Trajectory t5(5, std::vector<SamplePoint>{SamplePoint(20.5, 13, 7), SamplePoint(19, 13, 8), SamplePoint(17.5, 13, 9),
                                          SamplePoint(16, 13, 10), SamplePoint(15, 13, 11), SamplePoint(14, 12.5, 12),
                                          SamplePoint(12.5, 11, 13), SamplePoint(12.5, 10, 14),
                                          SamplePoint(12.5, 8, 15), SamplePoint(12.5, 6, 16),
                                          SamplePoint(13.5, 4.5, 17), SamplePoint(14.5, 3.5, 18)});

Trajectory t6(6,
              std::vector<SamplePoint>{SamplePoint(2, 6, 8), SamplePoint(2, 4.5, 9), SamplePoint(2, 3.5, 10),
                                       SamplePoint(2.5, 2.5, 11), SamplePoint(3.5, 2.5, 12), SamplePoint(4.5, 2.5, 13),
                                       SamplePoint(5.5, 2.5, 14), SamplePoint(6.5, 2.5, 15), SamplePoint(7.5, 2.5, 16),
                                       SamplePoint(9, 2.5, 17), SamplePoint(11, 2.5, 18), SamplePoint(12, 2.5, 19),
                                       SamplePoint(14.5, 2.5, 20), SamplePoint(16, 2.5, 21)});

Trajectory t7(7, std::vector<SamplePoint>{SamplePoint(15.5, 6.5, 9), SamplePoint(15.5, 6, 10), SamplePoint(15.5, 5, 11),
                                          SamplePoint(16.5, 5, 12), SamplePoint(17.5, 5, 13), SamplePoint(17.5, 4, 14),
                                          SamplePoint(18, 3, 15), SamplePoint(18, 2.5, 16), SamplePoint(20, 2.5, 17),
                                          SamplePoint(22, 2.5, 18)});

