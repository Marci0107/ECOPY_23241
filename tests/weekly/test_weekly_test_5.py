import random
from pathlib import Path

import pandas as pd
import src.weekly.weekly_test_5 as wt
from pandas.testing import assert_frame_equal, assert_series_equal
from pytest import approx

from src.weekly.distributions import UniformDistribution, LogisticDistribution, CauchyDistribution, \
    ChiSquaredDistribution
from src.weekly.weekly_test_2 import LaplaceDistribution


class TestWeekly5:
    path_to_datalib = Path.cwd().parent.parent.joinpath('data')
    ogdata = pd.read_csv(path_to_datalib.joinpath('chipotle.tsv'), sep='\t')
    refdata = pd.read_csv(path_to_datalib.joinpath('weekly5').joinpath('cleaned.csv'))

    def test_change_price_to_float(self):
        # Arrange
        expected = 'float64'

        # Act
        result = wt.change_price_to_float(self.ogdata)

        # Assert
        assert result.item_price.dtype == expected

    def test_number_of_observations(self):
        # Arrange
        expected = 4622

        # Act
        result = wt.number_of_observations(self.refdata)

        # Assert
        assert result == expected

    def test_items_and_prices(self):
        # Arrange
        expected = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('items_and_prices.csv'))
        side_effect = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('cleaned.csv'))

        # Act
        result = wt.items_and_prices(self.refdata)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
        assert_frame_equal(self.refdata, side_effect)

    def test_sorted_by_price(self):
        # Arrange
        expected = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('sorted_by_price.csv'))
        data = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('items_and_prices.csv'))
        side_effect = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('items_and_prices.csv'))

        # Act
        result = wt.sorted_by_price(data)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
        assert_frame_equal(data, side_effect)

    def test_avg_price(self):
        # Arrange
        expected = 7.464335785374297

        # Act
        result = wt.avg_price(self.refdata)

        # Assert
        assert result == approx(expected)

    def test_unique_items_over_ten_dollars(self):
        # Arrange
        expected = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('unique_items_over_ten_dollars.csv'))
        side_effect = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('cleaned.csv'))

        # Act
        result = wt.unique_items_over_ten_dollars(self.refdata)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
        assert_frame_equal(self.refdata, side_effect)

    def test_items_starting_with_s(self):
        # Arrange
        expected = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('items_starting_with_s.csv'))
        expected = expected.squeeze()
        side_effect = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('cleaned.csv'))

        # Act
        result = wt.items_starting_with_s(self.refdata)

        # Assert
        assert_series_equal(result.reset_index(drop=True), expected.reset_index(drop=True), check_index=False)
        assert_frame_equal(self.refdata, side_effect)

    def test_first_three_columns(self):
        # Arrange
        expected = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('first_three_columns.csv'))
        side_effect = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('cleaned.csv'))

        # Act
        result = wt.first_three_columns(self.refdata)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
        assert_frame_equal(self.refdata, side_effect)

    def test_every_column_except_last_two(self):
        # Arrange
        expected = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('every_column_except_last_two.csv'))
        side_effect = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('cleaned.csv'))

        # Act
        result = wt.every_column_except_last_two(self.refdata)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
        assert_frame_equal(self.refdata, side_effect)

    def test_sliced_view(self):
        # Arrange
        expected = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('sliced_view.csv'))
        side_effect = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('cleaned.csv'))

        columns_to_keep = ['quantity', 'item_name', 'item_price']
        column_to_filter = 'order_id'
        rows_to_keep = [1, 2, 4, 6]

        # Act
        result = wt.sliced_view(self.refdata, columns_to_keep, column_to_filter, rows_to_keep)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
        assert_frame_equal(self.refdata, side_effect)

    def test_generate_quartile(self):
        # Arrange
        expected = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('generate_quartile.csv'))
        side_effect = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('cleaned.csv'))

        # Act
        result = wt.generate_quartile(self.refdata)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
        assert_frame_equal(self.refdata, side_effect)

    def test_average_price_in_quartiles(self):
        # Arrange
        expected = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('average_price_in_quartiles.csv'))
        data = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('generate_quartile.csv'))
        side_effect = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('generate_quartile.csv'))
        expected = expected.squeeze()

        # Act
        result = wt.average_price_in_quartiles(data)

        # Assert
        assert_series_equal(result.reset_index(drop=True), expected.reset_index(drop=True), check_index=False)
        assert_frame_equal(data, side_effect)

    def test_minmaxmean_price_in_quartile(self):
        # Arrange
        expected = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('minmaxmean_price_in_quartile.csv'))
        data = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('generate_quartile.csv'))
        side_effect = pd.read_csv(self.path_to_datalib.joinpath('weekly5').joinpath('generate_quartile.csv'))

        # Act
        result = wt.minmaxmean_price_in_quartile(data)

        # Assert
        assert_frame_equal(result.reset_index(drop=True), expected.reset_index(drop=True))
        assert_frame_equal(data, side_effect)

    def test_gen_uniform_mean_trajectories(self):
        # Arrange
        dist = UniformDistribution(random, 0, 1)
        expected = [
            approx(
                [0.6394267984578837, 0.33221877684027534, 0.31315562401655667, 0.29066940254962315, 0.379829764872501,
                 0.429308051964236, 0.49543255421289445, 0.44437083901495966, 0.4418765035338831, 0.4006685751243018,
                 0.384120338731511, 0.3942232511791653, 0.3659396141410652, 0.35400375960860686, 0.37372913815333464,
                 0.38442990955645223, 0.3747834808790549, 0.38669915882332456, 0.4089481744998773, 0.38882570375878645,
                 0.40868253938135896, 0.4218396691816711, 0.4182923147180329, 0.4073417807636057, 0.4293366324213327,
                 0.42576962906330557, 0.4134354147787442, 0.4021240205664128, 0.41748161869679373, 0.423689765785797,
                 0.4360587498983319, 0.44523603229819086, 0.4479933674241457, 0.463438143793417, 0.4610123218909867,
                 0.46354088604049326, 0.4734290962624528, 0.4772472714230263, 0.48710521062527634, 0.48936138399106344,
                 0.49461041941115763, 0.48392503760745537, 0.4779709268642947, 0.4736849504265177, 0.46493177323756457,
                 0.4598852322185095, 0.4522494066268333, 0.4486186607202345, 0.4524363297721562, 0.45068424675611474,
                 0.44910575107691414, 0.44449808337873287, 0.44114864448572116, 0.4503246804714021, 0.4539194205582238,
                 0.4566910560065946, 0.4516813646415332, 0.4564649065951335, 0.45149774705558765, 0.45029704196728865,
                 0.4591368175192445, 0.4620539617504517, 0.463560244004804, 0.46701421286394573, 0.4727963314381898,
                 0.4773903250761906, 0.4736837242834729, 0.4671898495720106, 0.4649907654921131, 0.46217290992759297,
                 0.4586350216692653, 0.4653610590674012, 0.4709912860181584, 0.46887894270437885, 0.47136640567225224,
                 0.4703698990326261, 0.4761384404703913, 0.4759168175488145, 0.4732454675355137, 0.4704127430374946,
                 0.47153564910077433, 0.4689893803132397, 0.47038211055312285, 0.4754706911846628, 0.4745757478194362,
                 0.4716076665559228, 0.47765283827936167, 0.478015036636147, 0.47366553523769356, 0.4689261001286607,
                 0.46497800155967167, 0.4667439585177523, 0.4702421886881309, 0.46973067568931764, 0.4654548549573454,
                 0.4645815677859828, 0.47006135966695306, 0.47066394115095495, 0.47571863242835555,
                 0.47956924312641697]),
            approx([0.011481021942819636, 0.36610142065150714, 0.47130440344319635, 0.48772088518459605,
                    0.44354174613818537,
                    0.47644508821178916, 0.4243175289809445, 0.42562349419196455, 0.42874573998499144,
                    0.4812527587386003,
                    0.5171255025240179, 0.49598079820960733, 0.49633505319735277, 0.4736433980068369,
                    0.5029090274293692,
                    0.5258846238298316, 0.5125058101603494, 0.5195304593106634, 0.5242378146857956, 0.5056678873789875,
                    0.517898502269281, 0.5188748898988421, 0.5301684372350036, 0.5301761553583442, 0.5089919849891281,
                    0.5018829108358799, 0.4840160157081004, 0.49991182287083313, 0.5129742385588452, 0.5235972815855896,
                    0.5166268571925919, 0.5022924293582668, 0.5136777981414721, 0.5264210818813697, 0.5138277210295557,
                    0.5130544638708628, 0.5010587356167417, 0.5078888258704389, 0.5145028157021453, 0.5048500319220858,
                    0.5041288696337113, 0.5052163630589787, 0.4996312529631898, 0.508104020875055, 0.5062158857489624,
                    0.499815501394465, 0.5006555138920181, 0.5054320879586423, 0.49922227113070455, 0.49547215153410834,
                    0.5052697437914964, 0.5080506729039571, 0.5067308504701938, 0.5069316836288122, 0.4999148202149841,
                    0.49500022230099433, 0.49224733352637084, 0.49390356430104093, 0.48943256715350764,
                    0.48494564744180846,
                    0.4781595398773367, 0.4806263691901232, 0.4766313757714094, 0.4833311982282019, 0.4891204936439795,
                    0.4827831732840504, 0.47912976225546744, 0.48192164484430294, 0.478042154446131,
                    0.47310315007868664,
                    0.479615982339278, 0.4808858033251948, 0.4807732721332304, 0.4848793012164402, 0.4891808705037762,
                    0.48524967371243544, 0.48020657164121966, 0.47957637434333705, 0.47886754204810394,
                    0.47871950612296105,
                    0.4818103251765029, 0.4841463522754883, 0.49017067587898805, 0.4855069520131901, 0.4845318264848263,
                    0.48284311461168833, 0.4871974757811263, 0.4844867809872531, 0.48118028803729307, 0.480818435368358,
                    0.48017077827457827, 0.47797919530058225, 0.47552572489823297, 0.4802889150511881,
                    0.47989777641963305,
                    0.4838712277565309, 0.48455632141316307, 0.48012807659797646, 0.4853720603536811,
                    0.4888786156009438])]

        # Act
        result = wt.gen_uniform_mean_trajectories(dist, 2, 100)

        # Assert
        assert result == expected

    def test_gen_logistic_mean_trajectories(self):
        # Arrange
        dist = LogisticDistribution(random, 1, 3.3)
        expected = [
            approx(
                [2.8904946145457684, -4.098901515545551, -3.4654465387695006, -3.377903426107615, -1.8240357616304899,
                 -0.9471083761183583, 0.32727293454411793, -0.5586703887138117, -0.5009441208125516,
                 -1.5002694153524607,
                 -1.6550585870181296, -1.4279126625795608, -2.1555953087785062, -2.2586811663225483,
                 -1.9053588733500395,
                 -1.6865968885099867, -1.77375187238161, -1.5534839388044168, -1.1678881983536795, -1.8893817435675615,
                 -1.5281668436062865, -1.2874821512316201, -1.2830345580722449, -1.4205931974240318,
                 -0.9135407430883814,
                 -0.9260612530448923, -1.1334607307151527, -1.3205884796490757, -1.0454036093208519,
                 -0.9309118728292983,
                 -0.7162434582039354, -0.560180685491418, -0.49838580696481516, -0.10597511376234447,
                 -0.12112020246416336,
                 -0.07082705551820775, 0.0991592377189124, 0.16483376241488837, 0.3410554990964847, 0.3832619379514542,
                 0.4682614793992437, 0.2423765964692838, 0.16635118963079834, 0.11792087355229608, -0.04179033319010716,
                 -0.10470015544528907, -0.23469130869158716, -0.27459289165260814, -0.2110899713366042,
                 -0.22346198971037112, -0.23385986814236612, -0.29440233666022275, -0.3328670889694353,
                 -0.14356862612822857, -0.08615158984270511, -0.040611722135313846, -0.11368888893393517,
                 -0.03814857128476552, -0.11189720385181481, -0.12041789673290622, 0.14399298179919998,
                 0.18842371174747805,
                 0.21329020745631463, 0.26554653037239073, 0.36211795363326166, 0.43390813109659765, 0.3825782994125726,
                 0.22635399859570718, 0.20051312382097797, 0.16450323797314323, 0.11496464154577203, 0.2557888916951906,
                 0.3545172860099162, 0.3285302927228689, 0.36577665721730157, 0.35572423544100057, 0.4656830198653027,
                 0.46555392013420244, 0.42967988558166764, 0.3907458101643826, 0.4083188762910785, 0.374012154870661,
                 0.3951369870728685, 0.48771600790697434, 0.4779042904620828, 0.4352568467790298, 0.6694919957582943,
                 0.674676885525653, 0.5929555949152393, 0.4872263310626661, 0.41691297298747615, 0.4419488116220892,
                 0.4954092220850274, 0.48975686748350067, 0.401663378686081, 0.3913038983080557, 0.5863387674633459,
                 0.5944857656012578, 0.7157091252673946, 0.7786708088119227]),
            approx(
                [-13.703191035273809, -4.787321451784004, -2.0204068567513658, -1.14307998253666, -1.3815856727956175,
                 -0.6659071415372833, -1.4061269230379023, -1.21361545333422, -1.0357248767654483, 0.16703339772952752,
                 0.8288770461895401, 0.5603196888827897, 0.5947363817231628, 0.2640997715056373, 0.8293131100523717,
                 1.2330010616651381, 1.0533799230656875, 1.1550624247295804, 1.223840850420283, 0.9300857130095121,
                 1.1167212471097603, 1.1350921888743428, 1.3096683582803563, 1.3134805837537973, 0.3154318528466261,
                 0.2485062986096138, -0.20263302810879336, 0.14355573323891851, 0.39844210815190756, 0.5942165579478146,
                 0.5208924593556554, 0.24825587831218293, 0.468407519909037, 0.7637661597766767, 0.5472565432780476,
                 0.5546945754039594, 0.3349406012154626, 0.4528302219520548, 0.5671241640206741, 0.4199374000937211,
                 0.42612090775284195, 0.45548933007753445, 0.3898848143959341, 0.5479494070655826, 0.5352686129586675,
                 0.4510976301057274, 0.47383555970573654, 0.5531535792202832, 0.46939367484058236, 0.4277266951991874,
                 0.7834277191366397, 0.8268437379159906, 0.8146147618048953, 0.8223459093501484, 0.7065986895681289,
                 0.6388550125822174, 0.6062950462331954, 0.6333938850213129, 0.5720602686838152, 0.509650596813637,
                 0.3785733546717685, 0.417176037773702, 0.36282122422378615, 0.4892544086741718, 0.5891193344213114,
                 0.46666512005772165, 0.4173112215037974, 0.4600237859790789, 0.40569601367882846, 0.325525878091055,
                 0.4593403226845465, 0.4799627779877367, 0.48213996820301824, 0.5467896211299177, 0.6159208400328473,
                 0.5581291021540975, 0.4682190409776054, 0.46329366838341446, 0.4572174276433148, 0.4585533709499335,
                 0.5055687404162045, 0.5407125245250044, 0.7104344059292574, 0.626866557898091, 0.6159383490573648,
                 0.5948328818476035, 0.6688753773037222, 0.631170979595452, 0.5816010141566083, 0.5786865078712391,
                 0.5718912977303475, 0.5424079588596958, 0.5083086480168213, 0.6008688809601405, 0.5971340870089693,
                 0.6641179420439586, 0.6744523042915346, 0.5790395129307709, 0.8245908323324117, 0.8801007161126538])]

        # Act
        result = wt.gen_logistic_mean_trajectories(dist, 2, 100)

        # Assert
        assert result == expected

    def test_gen_laplace_mean_trajectories(self):
        # Arrange
        dist = LaplaceDistribution(random, 1, 3.3)
        expected = [
            approx([2.0788132521590237, -3.4028419332355315, -2.59273138909601, -2.359904243068386, -1.2652293695709382,
                    -0.6478769357778592, 0.3107719168577972, -0.3247034341085735, -0.239769786736851,
                    -1.0464565035779765,
                    -1.1085722491158345, -0.9298965995136411, -1.5267624687405186, -1.5636361586010115,
                    -1.314331220815166,
                    -1.1502604668172116, -1.182752974714451, -1.0254343075109817, -0.7512982905320668,
                    -1.3803278411560922,
                    -1.1183502463843258, -0.9463651847593707, -0.9169691157769411, -0.9977083564260392,
                    -0.5932944604090542,
                    -0.5822410494209145, -0.7295528329133236, -0.8614017912200893, -0.6620968449179642,
                    -0.5811183652504645,
                    -0.42871057591369033, -0.3206215665218606, -0.2730812068256909, 0.04807189975533663,
                    0.04903001759433171,
                    0.08552052722386638, 0.20614255495523048, 0.25052858722142907, 0.3784962667999937,
                    0.4078995328950782,
                    0.4646922733749549, 0.28966834327890884, 0.24588910729838961, 0.2220150005264165,
                    0.10472331255643616,
                    0.06934361330142094, -0.023158640514808475, -0.04220471902216968, 0.00038597715003730904,
                    -0.00042300425123364515, -0.000258497691942116, -0.0362248273424901, -0.05574059046739832,
                    0.09006605976704281, 0.12767496047457966, 0.15776250826145974, 0.11046768521311855,
                    0.16067960462846215,
                    0.112351264557487, 0.11197250946551505, 0.3356453424252768, 0.3638455985098848, 0.38027744830662963,
                    0.4137212117543222, 0.4815021858579319, 0.5295062959291527, 0.4980773744127412, 0.3722092255067756,
                    0.3592790308073094, 0.3389873026688066, 0.3081939150496835, 0.41725948027800164, 0.4884076476391446,
                    0.4746711480037248, 0.49805831599446215, 0.494496907292156, 0.5767753889190216, 0.5785679311683211,
                    0.5573633889662672, 0.5337437767531179, 0.5448349291156411, 0.5244913057513202, 0.5375889662018181,
                    0.6054756591506676, 0.601395671844337, 0.5744092082719722, 0.7808466315300777, 0.7840583767191599,
                    0.7232750574021054, 0.6397435712302155, 0.5886786481331435, 0.6037032988577318, 0.6390999348721942,
                    0.6369984621576313, 0.5691528414813987, 0.5643532364390272, 0.7341550055705068, 0.7388878754708968,
                    0.8365259917979598, 0.8803528998277323]),
            approx([-11.45391188387518, -4.265996354250982, -2.0138532611396567, -1.1970159495659287, -1.17210224630997,
                    -0.6279365875336521, -1.1025716157024321, -0.8974184337258148, -0.7222047060473664,
                    0.2360668912908263,
                    0.7234577369666628, 0.5702345588609155, 0.6035911785788088, 0.3893162615864375, 0.8138034175983513,
                    1.1040990201959298, 0.9978063514379834, 1.057619765986252, 1.097282955881904, 0.8968572999269494,
                    1.0187595063561876, 1.030211689235655, 1.1457980092438749, 1.1483344670817643, 0.24831149679196315,
                    0.22221628399790602, -0.1456353280534968, 0.1254927154331664, 0.31683853483413715,
                    0.45936265584542907,
                    0.4250578169254452, 0.22074334428572814, 0.3854237562161616, 0.6212376799974093,
                    0.46571130967275604,
                    0.4779475782425541, 0.31569205036727666, 0.3976577452134963, 0.4772899286307411, 0.3781969351331268,
                    0.3892822130644389, 0.4120671532659522, 0.3770331919388095, 0.49363903126445796, 0.4926514398999512,
                    0.4420586986848717, 0.45967688321032446, 0.5132788622835172, 0.46188906275700586,
                    0.44146531479650064,
                    0.7523608829691927, 0.779736207777164, 0.7756632538606226, 0.7820044608728525, 0.7008410435199438,
                    0.6590489276944165, 0.6423757885184348, 0.6595986774019403, 0.6219631192688243, 0.5831641228644008,
                    0.4843961126417968, 0.5088977670187154, 0.4757761652729758, 0.5698270885621393, 0.6409405758873565,
                    0.548683900245035, 0.5188580324168183, 0.5459482752900813, 0.5119948772172606, 0.45629247387274624,
                    0.5591466665581722, 0.5722936628017882, 0.575611723495521, 0.6189043353571421, 0.665983450232849,
                    0.6284584654872275, 0.562971796087757, 0.5622970529729945, 0.560908893590515, 0.5635832091038833,
                    0.5939357419949355, 0.616022124783637, 0.7579124115886378, 0.6969399687315545, 0.6920957491892568,
                    0.6807985664294318, 0.7332083321882331, 0.7100449429229968, 0.6774669000790046, 0.6770742377161076,
                    0.67446226935029, 0.6570160246356003, 0.6360809967005636, 0.7057508939237273, 0.704654014198336,
                    0.7518215897749189, 0.7579891629630396, 0.6833165416189905, 0.9047335779864256,
                    0.9424782684594942])]

        # Act
        result = wt.gen_laplace_mean_trajectories(dist, 2, 100)

        # Assert
        assert result == expected

    def test_gen_cauchy_mean_trajectories(self):
        # Arrange
        dist = CauchyDistribution(random, 2, 4)
        expected = [
            approx([3.873466521685142, -22.46470318913162, -15.448363998453912, -12.270538078559479, -8.681698830978501,
                    -6.487969675002172, -3.6534369672253684, -4.731662520048834, -4.094958083909323, -7.745988173728804,
                    -7.30339956807243, -6.522507696963586, -9.549271905302117, -9.120610977739583, -8.243485384257374,
                    -7.567734169569807, -7.288547722000851, -6.7085113178990605, -5.941577709891198,
                    -15.339157297957456,
                    -14.240974744691428, -13.37226690593232, -12.799338816616855, -12.49633487220421,
                    -10.733352096572437,
                    -10.330341568894877, -10.367645797111031, -10.381552502717371, -9.689085359532575,
                    -9.25439435755125,
                    -8.705120596119126, -8.260570307296884, -7.935788431416294, -6.253929562998604, -6.063961740425098,
                    -5.821633596301504, -5.428215111067672, -5.191624340836949, -4.78619512208438, -4.591749425303984,
                    -4.357929199615616, -4.8635257167394785, -4.810838074459351, -4.72684300668737, -4.924498643077105,
                    -4.870872662249522, -4.983837343149668, -4.908178382755327, -4.730125073415439, -4.631694745859074,
                    -4.535548873955409, -4.509351639503932, -4.454355236624416, -3.9675356032510503, -3.82254220331975,
                    -3.6930724288053614, -3.71089120716411, -3.5519617100163723, -3.5781191617889907, -3.51167772992068,
                    -1.4297274240677305, -1.344050313375308, -1.2794878226717423, -1.1872993360804278,
                    -1.023910834151825,
                    -0.9066803795530348, -0.9314247821217881, -1.4696380868672734, -1.457315314505441,
                    -1.4590286283199723,
                    -1.4824792280165935, -1.127687065994037, -0.9509319134945108, -0.9466396879198092,
                    -0.879018439654715,
                    -0.8590397837573992, -0.6330742006946516, -0.6059832822823567, -0.6191034127498796,
                    -0.637435505898464,
                    -0.5952341583445782, -0.6086085636618119, -0.5640628436757247, -0.3903223930693142,
                    -0.3775893378488693,
                    -0.4064107569604709, 5.564509355708516, 5.525364325418628, 5.332689466783554, 4.997596895516729,
                    4.8421400978084925, 4.829649244626736, 4.855426342761313, 4.8144309241212335, 4.576642618436447,
                    4.533550291431008, 7.891503912693236, 7.835130253717879, 8.219649954357408, 8.243001078426241]),
            approx(
                [-108.85140003691517, -51.76350255021666, -32.986091715768744, -24.122897972934535, -19.617920520678275,
                 -15.69876030229757, -14.733609985344717, -12.745838375622505, -11.172484886652873, -7.117728952628428,
                 -5.404251724703204, -5.0936387940501175, -4.547407708189648, -4.53419239345161, -3.1515952527535194,
                 -2.2493126593865647, -2.172066989421133, -1.8366100502324523, -1.5596575454120674, -1.7656899522337954,
                 -1.3803013746374582, -1.2040426037420766, -0.8563490479165413, -0.7213930089940807, -89.66634356676468,
                 -86.23555399806838, -85.38574461656091, -81.6341138755338, -78.40585078428191, -75.49748497603939,
                 -73.08673732654252, -71.41957655113438, -68.8941001647275, -66.10963062350285, -64.57805498087016,
                 -62.73355797561991, -61.4733335122428, -59.69046294660561, -57.99534441833699, -56.72978893520642,
                 -55.30494619602381, -53.92551871571364, -52.70953966048631, -51.25157764244313, -50.090100320273855,
                 -49.06851226671531, -47.971387894408835, -46.85688116352577, -45.97130946348186, -45.065622941410375,
                 -38.996326797166226, -38.16877874389616, -37.42574165738146, -36.69154108474071, -36.170066946390946,
                 -35.57225192406749, -34.952220335346766, -34.295473509534695, -33.75714029766188, -33.241662153718686,
                 -32.953050181105006, -32.361106871574684, -31.88819456359753, -31.154574141832217, -30.514115754578,
                 -30.28922403470253, -29.871674967357226, -29.368440220541007, -28.98654141469696, -28.67333799955189,
                 -27.967042422624107, -27.53822407104732, -27.13830692410056, -26.677240603709905, -26.217712042098285,
                 -25.923657249640154, -25.726297994490576, -25.382117847908358, -25.04790403320675, -24.71500359530805,
                 -24.34190617563079, -23.9911143655249, -22.709999746505595, -22.56490657507733, -22.29077070452442,
                 -22.034022728530033, -21.65871581851021, -21.435706699779157, -21.2384237029301, -20.987456883278423,
                 -20.745856908378215, -20.534923671502003, -20.335675755202065, -19.92497562741476, -19.701790011507,
                 -19.386200539212826, -19.159149719344907, -19.19789696929568, -1.0598498982419022,
                 -0.9585949052023325])]

        # Act
        result = wt.gen_cauchy_mean_trajectories(dist, 2, 100)

        # Assert
        assert result == expected

    def test_gen_chi2_mean_trajectories(self):
        # Arrange
        dist = ChiSquaredDistribution(random, 3)
        expected = [
            approx([3.208561989143851, 1.712210957436818, 1.5806562533060504, 1.4607871617285357, 1.9648192900357102,
                    2.2174321206511536, 2.76911166787526, 2.4887688639535033, 2.431435340491424, 2.2126855170852733,
                    2.1099181305187984, 2.1336397918820142, 1.986817681955173, 1.916357156393525, 2.00741915590777,
                    2.045323989333547, 1.9891112350114046, 2.0385236573707384, 2.1815432119156477, 2.076748902272204,
                    2.2022146369163726, 2.2680089101738625, 2.238896088865934, 2.179802713188699, 2.4190754922948328,
                    2.3868925544744295, 2.3189456323317827, 2.2564802741299004, 2.36068677593076, 2.3809971702485027,
                    2.456691517091702, 2.5024100933091518, 2.5042873762039632, 2.7008882079468868, 2.6742939201615537,
                    2.6737530541566716, 2.7370696389103744, 2.745739757311874, 2.8165101136782567, 2.8162268687191974,
                    2.83784416751423, 2.7781477239043246, 2.739600286460285, 2.708663295118337, 2.659448511730285,
                    2.6264367530865935, 2.583082120194926, 2.556975847237537, 2.5697442543593794, 2.5525083653416156,
                    2.5364220987600947, 2.5077307348431597, 2.4846346324875443, 2.57353187533805, 2.5861788194345006,
                    2.5936561359408903, 2.5636931525252855, 2.5869786919123157, 2.5576001482501716, 2.5445450612645883,
                    2.687160466670105, 2.6956345406549618, 2.6954408442204154, 2.708664641177208, 2.747124336848093,
                    2.7717429593881184, 2.747170323643216, 2.7105503378194373, 2.6928632893982134, 2.6727773188049624,
                    2.649929379897677, 2.7175455408342915, 2.759288424297354, 2.7420925236711735, 2.7498276198321387,
                    2.737972324558442, 2.7882485536838746, 2.780111860415374, 2.7610575856385324, 2.741525178279957,
                    2.7411285973628368, 2.7231373675911024, 2.7246599588896627, 2.766059113436122, 2.7554745443889876,
                    2.736051244005022, 2.8695753820833576, 2.8644327387485706, 2.8383621233786522, 2.810571615629833,
                    2.7865703065226715, 2.7902634020452832, 2.809179510956648, 2.800293632976225, 2.775218227395759,
                    2.764895049330413, 2.8743535287892588, 2.8707832533631485, 2.9329768203577693, 2.9585499921184515]),
            approx([0.12619119792858835, 1.9830495159330033, 2.4950759699251477, 2.5134447089588305,
                    2.2673549170843357, 2.4260068757398656, 2.1701039335989654, 2.153179150934443, 2.1503774181803266,
                    2.734507174205166, 3.009115610585933, 2.8640696789556386, 2.8259945471224053, 2.6896305129942486,
                    2.947575211928812, 3.1169841029988614, 3.016984915164197, 3.0274436102475586, 3.0262025844517013,
                    2.9153742837772945, 2.9780651341130846, 2.9600706702137938, 3.0226818573119467, 3.0022110328186535,
                    2.8827912074752438, 2.8306803777622775, 2.7325615901232543, 2.886094111004486, 2.9868863232761433,
                    3.0555814961404155, 3.003978792533485, 2.9223263294221478, 3.0093946779700853, 3.1468383742275527,
                    3.0718025256710786, 3.050147954668902, 2.9797398106099178, 3.0121849723312937, 3.0443264526544973,
                    2.9858877427602324, 2.967619651127765, 2.9598637105197816, 2.920693978243015, 2.9836898968527943,
                    2.9613543692942774, 2.9198889602423845, 2.912693800196731, 2.933710127245636, 2.8944493311492026,
                    2.866040368400452, 3.0628477953971203, 3.067066906846285, 3.0478948480987436, 3.0370220036465656,
                    2.9940817494170853, 2.9603899189561433, 2.9363282941436033, 2.9352291443855134, 2.9046286701693194,
                    2.8743645731445127, 2.834676348274649, 2.839781394584036, 2.8125616186755296, 2.868279150951078,
                    2.908325805566264, 2.8711200942667157, 2.8456193143719117, 2.8540952835350595, 2.8281523388323033,
                    2.7980843615096838, 2.8607168513263526, 2.8594155829780523, 2.8507047938646415, 2.8725275098107197,
                    2.8973210758783225, 2.871902868081835, 2.8420192229866874, 2.831438144334872, 2.8206693533764566,
                    2.81284585992067, 2.8264378558273138, 2.8341041811977536, 2.9246222209204085, 2.8966793057429703,
                    2.8847351090315936, 2.8697293094643195, 2.900022165913424, 2.8807824362100765, 2.859253491097065,
                    2.85084625715667, 2.841194923277775, 2.8247949451422856, 2.8074501077292053, 2.850487658720638,
                    2.8423307161799443, 2.870012633004235, 2.867691742658722, 2.842050058484244, 2.984736470520384,
                    3.00598124766088])]

        # Act
        result = wt.gen_chi2_mean_trajectories(dist, 2, 100)

        # Assert
        assert result == expected
