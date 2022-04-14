import numpy as np
import streamlit as st
import pandas as pd
#import matplotlib.pyplot as plt
from tensorflow.keras import utils #Используем для to_categorical
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler #
from  PIL import Image

st.set_page_config(layout="wide")

st.title("Подберите параметры квартиры")
with st.expander("Краткое описание:"):
    st.write("В данной задаче Вам предлагается оценить стоимость московской квартиры по ряду параметров (станция метро, этажность здания, площадь квартиры и т.д.). "
             "Нейроннная сеть обучена на уже готовой базе, собранной из разных сайтов по торговле недвижимостью.")

#data1 = pd.read_csv('/sysroot/home/user/Загрузки/PyProject/housereg3/venv/moscow21.csv')
#data2 = data1.values

def getRoomsCount(d, maxRoomCount):
    roomsCountStr = d[0]  # Получаем строку с числом комнат

    roomsCount = 0
    try:
        roomsCount = int(roomsCountStr)  # Пробуем превратить строку в число
        if (roomsCount > maxRoomCount):
            roomsCount = maxRoomCount  # Если число комнат больше максимального, то присваиваем максимальное
    except:  # Если не получается превратить строку в число
        if (roomsCountStr == roomsCountStr):  # Проверяем строку на nan (сравнение с самим собой)
            if ("Ст" in roomsCountStr):  # Еcть строка = "Ст", значит это Студия
                roomsCount = maxRoomCount + 1

    return roomsCount
# Превращаем число комнат в категорию
def getRoomsCountCategory(d, maxRoomCount):
    roomsCount = getRoomsCount(d, maxRoomCount)  # Получаем число комнат
    roomsCount = utils.to_categorical(roomsCount, maxRoomCount + 2)  # Превращаем в категорию
    # maxRoomCount+2 потому что 0 зарезервирован на неопознаное число комнат, а maxRoomCount+1 на "Студию"
    return roomsCount
# Получаем индекс станции метро
# allMetroNames - все уникальные названия метро в базе
def getMetro(d, allMetroNames):
    metroStr = d[1]  # Получаем строку метро
    metro = 0

    if (metroStr in allMetroNames):  # Если находим метро во всех названиях
        metro = allMetroNames.index(metroStr) + 1  # Присваиваем индекс
        # +1 так как 0 зарезервирован на неопознанное метро

    return metro
# Получаем тип метро
# 0 - внутри кольца
# 1 - кольцо
# 2 - 1-3 станции от конца
# 3 - 4-8 станций от кольца
# 4 - больше 8 станций от кольца
def getMetroType(d):
    metroTypeStr = d[1]  # Получаем строку метро
    metroTypeClasses = 5  # Число классов метро
    metroType = metroTypeClasses - 1  # Изначально считаем последний класс

    # Метро внутри кольца
    # Метро внутри кольца
    metroNamesInsideCircle = ["Площадь Революции м.", "Арбатская м.", "Смоленская м.", "Красные Ворота м.",
                              "Чистые пруды м.", "Лубянка м.", "Охотный Ряд м.", "Библиотека имени Ленина м.",
                              "Кропоткинская м.", "Сухаревская м.", "Тургеневская м.", "Китай-город м.",
                              "Третьяковская м.", "Трубная м.", "Сретенский бульвар м.", "Цветной бульвар м.",
                              "Чеховская м.", "Боровицкая м.", "Полянка м.", "Маяковская м.", "Тверская м.",
                              "Театральная м.", "Новокузнецкая м.", "Пушкинская м.", "Кузнецкий Мост м.",
                              "Китай-город м.", "Александровский сад м."]
    # Метро на кольце
    metroNamesCircle = ["Киевская м.", "Парк Культуры м.", "Октябрьская м.", "Добрынинская м.", "Павелецкая м.",
                        "Таганская м.", "Курская м.", "Комсомольская м.", "Проспект Мира м.", "Новослободская м.",
                        "Белорусская м.", "Краснопресненская м."]
    # Метро 1-3 станции от кольца
    metroNames13FromCircle = ["Бауманская м.", "Электрозаводская м.", "Семёновская м.", "Площадь Ильича м.",
                              "Авиамоторная м.", "Шоссе Энтузиастов м.", "Римская м.", "Крестьянская Застава м.",
                              "Дубровка м.", "Пролетарская м.", "Волгоградский проспект м.", "Текстильщики м.",
                              "Автозаводская м.", "Технопарк м.", "Коломенская м.", "Тульская м.", "Нагатинская м.",
                              "Нагорная м.", "Шаболовская м.", "Ленинский проспект м.", "Академическая м.",
                              "Фрунзенская м.", "Спортивная м.", "Воробьёвы горы м.", "Студенческая м.",
                              "Кутузовская м.", "Фили м.", "Парк Победы м.", "Выставочная м.", "Международная м.",
                              "Улица 1905 года м.", "Беговая м.", "Полежаевская м.", "Динамо м.", "Аэропорт м.",
                              "Сокол м.", "Деловой центр м.", "Шелепиха м.", "Хорошёвская м.", "ЦСКА м.",
                              "Петровский парк м.", "Савёловская м.", "Дмитровская м.", "Тимирязевская м.",
                              "Достоевская м.", "Марьина Роща м.", "Бутырская м.", "Фонвизинская м.", "Рижская м.",
                              "Алексеевская м.", "ВДНХ м.", "Красносельская м.", "Сокольники м.",
                              "Преображенская площадь м."]
    # Метро 4-8 станций от кольа
    metroNames48FromCircle = ["Партизанская м.", "Измайловская м.", "Первомайская м.", "Щёлковская м.", "Новокосино м.",
                              "Новогиреево м.", "Перово м.", "Кузьминки м.", "Рязанский проспект м.", "Выхино м.",
                              "Лермонтовский проспект м.", "Жулебино м.", "Партизанская", "Измайловская м.",
                              "Первомайская м.", "Щёлковская м.", "Новокосино м.", "Новогиреево м.", "Перово м.",
                              "Кузьминки м.", "Рязанский проспект м.", "Выхино м.", "Лермонтовский проспект м.",
                              "Жулебино м.", "Улица Дмитриевского м.", "Кожуховская м.", "Печатники м.", "Волжская м.",
                              "Люблино м.", "Братиславская м.", "Коломенская м.", "Каширская м.", "Кантемировская м.",
                              "Царицыно м.", "Орехово м.", "Севастопольская м.", "Чертановская м.", "Южная м.",
                              "Пражская м.", "Варшавская м.", "Профсоюзная м.", "Новые Черёмушки м.", "Калужская м.",
                              "Беляево м.", "Коньково м.", "Университет м.", "Багратионовская м.", "Филёвский парк м.",
                              "Пионерская м.", "Кунцевская м.", "Молодёжная м.", "Октябрьское Поле м.", "Щукинская м.",
                              "Спартак м.", "Тушинская м.", "Сходненская м.", "Войковская м.", "Водный стадион м.",
                              "Речной вокзал м.", "Беломорская м.", "Ховрино м.", "Петровско-Разумовская м.",
                              "Владыкино м.", "Отрадное м.", "Бибирево м.", "Алтуфьево м.", "Фонвизинская м.",
                              "Окружная м.", "Верхние Лихоборы м.", "Селигерская м.", "Ботанический сад м.",
                              "Свиблово м.", "Бабушкинская м.", "Медведково м.", "Преображенская площадь м.",
                              "Черкизовская м.", "Бульвар Рокоссовского м."]

    # Проверяем, в какую категорию попадает наша станция
    if (metroTypeStr in metroNamesInsideCircle):
        metroType = 0
    if (metroTypeStr in metroNamesCircle):
        metroType = 1
    if (metroTypeStr in metroNames13FromCircle):
        metroType = 2
    if (metroTypeStr in metroNames48FromCircle):
        metroType = 3

    # Превращаем результат в категорию
    metroType = utils.to_categorical(metroType, metroTypeClasses)
    return metroType

# Вычисляем растояние до метро
def getMetroDistance(d):
    metroDistanceStr = d[2]  # Получаем строку

    metroDistance = 0  # Расстояние до метро
    metroDistanceType = 0  # Тип расстояния - пешком или на транспорте

    # ЕСли строка не равна nan
    if (metroDistanceStr == metroDistanceStr):
        if (len(metroDistanceStr) > 0):
            # Определяем тип расстояния
            if (metroDistanceStr[-1] == "п"):
                metroDistanceType = 1  # Пешком
            elif (metroDistanceStr[-1] == "т"):
                metroDistanceType = 2  # На транспорте

            # Выбрасываем последний символ, чтобы осталось только число
            metroDistanceStr = metroDistanceStr[:-1]
            try:
                # Разделяем дистанции на категории
                metroDistance = int(metroDistanceStr)
                if (metroDistance < 3):
                    metroDistance = 1
                elif (metroDistance < 6):
                    metroDistance = 2
                elif (metroDistance < 10):
                    metroDistance = 3
                elif (metroDistance < 15):
                    metroDistance = 4
                elif (metroDistance < 20):
                    metroDistance = 5
                else:
                    metroDistance = 6
            except:  # Если в строке не число, то категория 0
                metroDistance = 0

    # Число классов дистанции
    metroDistanceClasses = 7

    # У нас 7 категорий дистанции по расстоянию
    # И 3 типа дистанции - неопознанный, пешком и транспортом
    # Мы создадим вектор длины 3*7 = 21
    # Будем преобразовывать индекс расстояния 0-6 в 0-20
    # Для типа "Пешком" - ничего не меняем
    if (metroDistanceType == 2):
        metroDistance += metroDistanceClasses  # Для типа "Транспортом" добавляем 7
    if (metroDistanceType == 0):
        metroDistance += 2 * metroDistanceClasses  # Для неопознанного типа добавляем 14

    # Превращаем в категории
    metroDistance = utils.to_categorical(metroDistance, 3 * metroDistanceClasses)
    return metroDistance
# Получаем 4 данных
# - этаж квартиры
# - этажность дома
# - индикатор, что последний этаж
# - тип дома
def getHouseTypeAndFloor(d):
    try:
        houseStr = d[3]  # Получаем строку типа дома и этажей
    except:
        houseStr = ""

    houseType = 0  # Тип дома
    floor = 0  # Этаж квартиры
    floors = 0  # Этажность дома
    isLastFloor = 0  # Индикатор последнего этажа

    # Проверяем строку на nan
    if (houseStr == houseStr):
        if (len(houseStr) > 1):

            try:
                slashIndex = houseStr.index("/")  # Ищем разделитель /
            except:
                print(houseStr)

            try:
                spaceIndex = houseStr.index(" ")  # Ищем разделитель " "
            except:
                print(houseStr)

            # Вытаскиваем строки
            floorStr = houseStr[:slashIndex]  # Строка этажа
            floorsStr = houseStr[slashIndex + 1:spaceIndex]  # Строка этажнгости дома
            houseTypeStr = houseStr[spaceIndex + 1:]  # Строка типа дома

            # Выбираем категорию этажа
            try:
                floor = int(floorStr)  # Превращаем строку в число
                floorSave = floor
                if (floorSave < 5):
                    floor = 2
                if (floorSave < 10):
                    floor = 3
                if (floorSave < 20):
                    floor = 4
                if (floorSave >= 20):
                    floor = 5
                if (floorSave == 1):  # Первый этаж выделяем в отдельную категорию
                    floor = 1

                if (floor == floors):  # Если этаж последний, включаем индикатор последнего этажа
                    isLastFloor = 1
            except:
                floor = 0  # Если строка не парсится в число, то категория этажа = 0 (отдельная)

            # Выбираем категорию этажности дома
            try:
                floors = int(floorsStr)  # Превращаем строку в число
                floorsSave = floors
                if (floorsSave < 5):
                    floors = 1
                if (floorsSave < 10):
                    floors = 2
                if (floorsSave < 20):
                    floors = 3
                if (floorsSave >= 20):
                    floors = 4
            except:
                floors = 0  # Если строка не парсится в число, то категория этажности = 0 (отдельная)

            # Определяем категорию типа дома
            if (len(houseTypeStr) > 0):
                if ("М" in houseTypeStr):
                    houseType = 1
                if ("К" in houseTypeStr):
                    houseType = 2
                if ("П" in houseTypeStr):
                    houseType = 3
                if ("Б" in houseTypeStr):
                    houseType = 4
                if ("?" in houseTypeStr):
                    houseType = 5
                if ("-" in houseTypeStr):
                    houseType = 6

        # Превращаем все категории в one hot encoding
        floor = utils.to_categorical(floor, 6)
        floors = utils.to_categorical(floors, 5)
        houseType = utils.to_categorical(houseType, 7)

    return floor, floors, isLastFloor, houseType
# Вычисляем тип балкона
def getBalcony(d):
    balconyStr = d[4]  # Полуаем строку
    # Выписываем все варианты балконов в базе
    balconyVariants = ['Л', 'Б', '2Б', '-', '2Б2Л', 'БЛ', '3Б', '2Л', 'Эрк', 'Б2Л', 'ЭркЛ', '3Л', '4Л', '*Л',
                       '*Б']
    # Проверяем на nan
    if (balconyStr == balconyStr):
        balcony = balconyVariants.index(balconyStr) + 1  # Находим индекс строки балкона во всех строках
    else:
        balcony = 0  # Индекс 0 выделяем на строку nan

    # Превращаем в one hot encoding
    balcony = utils.to_categorical(balcony, 16)

    return balcony
# Определяем тип санузла
def getWC(d):
    wcStr = d[5]  # Получаем строку
    # Выписываем все варианты санузлов в базе
    wcVariants = ['2', 'Р', 'С', '-', '2С', '+', '4Р', '2Р', '3С', '4С', '4', '3', '3Р']
    # Проверяем на nan
    if (wcStr == wcStr):
        wc = wcVariants.index(wcStr) + 1  # Находим индекс строки санузла во всех строках
    else:
        wc = 0  # Индекс 0 выделяем на строку nan

    # Превращаем в one hot encoding
    wc = utils.to_categorical(wc, 14)

    return wc
# Определяем площадь
def getArea(d):
    areaStr = d[6]  # Поулачем строку площади

    if ("/" in areaStr):
        slashIndex = areaStr.index("/")  # Находим разделитель /
        try:
            area = float(areaStr[:slashIndex])  # Берём число до разделителя и превращаем в число
        except:
            area = 0  # Если не получается, возвращаем 0
    else:
        area = 0  # Или если нет разделителя, возвращаем 0

    return area
# Полуаем цену
def getCost(d):
    costStr = d[7]  # Загружаем строку

    try:
        cost = float(costStr)  # Пробуем превратить в число
    except:
        cost = 0  # Если не получается, возвращаем 0

    return cost
# Объединяем все числовые параметры вместе
def getAllParameters(d, allMetroNames):
    # Загружаем все данные по отдельности
    roomsCountType = getRoomsCountCategory(d, 30)
    metro = getMetro(d, allMetroNames)
    metroType = getMetroType(d)
    metroDistance = getMetroDistance(d)
    floor, floors, isLastFloor, houseType = getHouseTypeAndFloor(d)
    balcony = getBalcony(d)
    wc = getWC(d)
    area = getArea(d)

    # Объединяем в один лист
    out = list(roomsCountType)
    out.append(metro)
    out.extend(metroType)
    out.extend(metroDistance)
    out.extend(floor)
    out.extend(floors)
    out.append(isLastFloor)
    out.extend(houseType)
    out.extend(balcony)
    out.extend(wc)
    out.append(area)

    return out
# Генерируем обучающаюу выборку - xTrain
def getXTrain(data):
    # Получаем строку во всеми вариантами метро
    allMertroNames = list(data1["Метро / ЖД станции"].unique())

    # Всевращаем все строки в data1 в векторы параметров и записываем в xTrain
    xTrain = [getAllParameters(d, allMertroNames) for d in data]
    xTrain = np.array(xTrain)

    return xTrain
# Генерируем обучающую выборку - yTrain
def getYTrain(data):
    # Зашружаем лист всех цен квартир по всем строкам data1
    costList = [getCost(d) for d in data]
    yTrain = np.array(costList)

    return yTrain

#oneRoomMask = [getRoomsCount(d, 30) == 1 for d in data2] #Делаем маску однокомнатных квартир, принцип (getRoomsCount(d, 30) == 1)
#data2 = data2[oneRoomMask] #В data1 оставляем только однокомнатные квартиры

#xTrain = getXTrain(data2)
#yTrain = getYTrain(data2)

#Нормируем размер квартиры в xTrain
#xScaler = StandardScaler() #Создаём нормировщик нормальным распределением
#xScaler.fit(xTrain[:,-1].reshape(-1, 1)) #Обучаем его на площадях квартир (последня колонка в xTrain)
#xTrainScaled = xTrain.copy()
#xTrainScaled[:,-1] = xScaler.transform(xTrain[:,-1].reshape(-1, 1)).flatten() #Нормируем данные нормировщиком

#yScaler = StandardScaler() #Делаем нормальный нормировщик
#yScaler.fit(yTrain.reshape(-1, 1)) #Обучаем на ценах квартир
#yTrainScaled = yScaler.transform(yTrain.reshape(-1, 1))

#splitVal = 0.05 #Процент, который выделяем в проверочную выборку
#valMask = np.random.sample(xTrainScaled.shape[0]) < splitVal

model = load_model('/sysroot/home/user/Загрузки/PyProject/housereg3/venv/housemodel1.h5')

#Функция для подсчета MAE
def mae(inp, true_out):
  pred = model.predict(inp) #Получаем выход сети на проверочной выборке
  predUnscaled = yScaler.inverse_transform(pred).flatten() #Делаем обратное нормирование выхода к изначальным величинам цен квартир
  yTrainUnscaled = yScaler.inverse_transform(true_out).flatten() #Делаем такое же обратное нормирование yTrain к базовым ценам
  delta = predUnscaled - yTrainUnscaled #Считаем разность предсказания и правильных цен
  absDelta = abs(delta) #Берём модуль отклонения
  return round(sum(absDelta) / (1e+6 * len(absDelta)),3), predUnscaled, yTrainUnscaled


col11, col12, col13, col14 = st.columns(4)

with col11:
    with st.container():
        st.subheader("Расположение")
        option1 = st.selectbox('Какое метро?',["Площадь Революции м.", "Арбатская м.", "Смоленская м.", "Красные ворота м.","Чистые пруды м.", "Лубянка м."])
        minutes1 = st.text_input('Сколько минут до метро?')
        tran_pesh = st.selectbox('пешком или на транспорте?', ["пешком", "на транспорте"])
        if tran_pesh == "пешком":
            tran_pesh = "п"
        elif tran_pesh == "на транспорте":
            tran_pesh = "т"
        option2 = minutes1 + tran_pesh

with col12:
    with st.container():
        st.subheader("информация о доме")
        etazhei = st.text_input('Сколько всего этажей в доме?')
        house_type = st.selectbox('Какой тип дома? (монолитнокирпичный/кирпичный/панельный и т.д.)',['монолитнокирпичный', 'кирпичный', 'панельный'])
        if house_type == "монолитнокирпичный":
            house_type = "М"
        elif house_type == "кирпичный":
            house_type = "К"
        elif house_type == "панельный":
            house_type = "П"

with col13:
    with st.container():
        st.subheader("информация о квартире")
        option = st.selectbox('Сколько комнат в квартире?', ["1", "2", "3", "4", "5"])
        etazh = st.text_input('На каком этаже квартира?')
        option4 = st.selectbox('Какой тип балкона?',
                                       ['лоджия', 'один балкон', 'два балкона', 'нет балкона', 'два балкона и две лоджии', 'балкон и лоджия', 'три балкона',
                                        'две лоджии','эркер', 'балкон и две лоджии', 'эркер и лоджия', 'три лоджии', 'четыре лоджии','пристроенная лоджия',
                                        'пристроенный балкон'])
        if option4 == "лоджия":
            option4 = "Л"
        elif option4 == "один балкон":
            option4 = "Б"
        elif option4 == "два балкона":
            option4 = "2Б"
        elif option4 == "нет балкона":
            option4 = "-"
        elif option4 == "два балкона и две лоджии":
            option4 = "2Б2Л"
        elif option4 == "балкон и лоджия":
            option4 = "БЛ"
        elif option4 == "три балкона":
            option4 = "3Б"
        elif option4 == "две лоджии":
            option4 = "2Л"
        elif option4 == "эркер":
            option4 = "Эрк"
        elif option4 == "балкон и две лоджии":
            option4 = "Б2Л"
        elif option4 == "эркер и лоджия":
            option4 = "ЭркЛ"
        elif option4 == "три лоджии":
            option4 = "3Л"
        elif option4 == "четыре лоджии":
            option4 = "4Л"
        elif option4 == "пристроенная лоджия":
            option4 = "*Л"
        elif option4 == "пристроенный балкон":
            option4 = "*Б"

        option5 = st.selectbox("Какой вариант санузла?",
                                       ['два', 'раздельный', 'совмещенный', 'на ремонте', 'два санузла', 'работает', 'четыре раздельных',
                                        'два раздельных', 'три совмещенных', 'четыре совмещенных', '4', '3', '3Р'])
        if option5 == "два":
            option5 = "2"
        elif option5 == "раздельный":
            option5 = "Р"
        elif option5 == "совмещенный":
            option5 = "С"
        elif option5 == "на ремонте":
            option5 = "-"
        elif option5 == "два санузла":
            option5 = "2С"
        elif option5 == "работает":
            option5 = "+"
        elif option5 == "четыре раздельных":
            option5 = "4Р"
        elif option5 == "два раздельных":
            option5 = "2Р"
        elif option5 == "три совмещенных":
            option5 = "3С"
        elif option5 == "четыре совмещенных":
            option5 = "4С"
        elif option5 == "четыре":
            option5 = "4"
        elif option5 == "три":
            option5 = "3"
        elif option5 == "три раздельных":
            option5 = "3Р"


        sum_area = st.text_input("Общая площадь")
        kitchen_area = st.text_input("Площадь кухни")
        host_area = st.text_input("Площадь гостинной")

        option6 = str(sum_area) + "/" + str(kitchen_area) + "/" + str(host_area)

with col14:
    with st.container():
        st.subheader("результат")

        option3 = etazh + "/" + etazhei + " " + house_type

        allStations1 = ['Шелепиха м.', 'Пятницкое шоссе м.', 'Планерная м.', 'Шаболовская м.', 'Бабушкинская м.',
                        'Улица Дмитриевского м.',
                        'Бульвар Рокоссовского м.', 'Марьина Роща м.', 'Марьино м.', 'Печатники м.',
                        'Красногвардейская м.', 'Площадь Ильича м.',
                        'Славянский бульвар м.', 'Раменки м.', 'Ботанический сад м.', 'Маяковская м.',
                        'Хорошево м. (МЦК)', 'Проспект Вернадского м.',
                        'Серпуховская м.', 'Улица Академика Янгеля м.', 'Крылатское м.', 'Первомайская м.',
                        'Измайловская м.',
                        'Мичуринский проспект м.', 'Бульвар Дмитрия Донского м.', 'Коптево м. (МЦК)',
                        'Ростокино м. (МЦК)', 'Саларьево м.', 'Октябрьское поле м.', 'Фрунзенская м.', 'Спортивная м.',
                        'Царицыно м.',
                        'Арбатская м.', 'Юго-Западная м.', 'Белокаменная м. (МЦК)', 'Молодежная м.', 'Медведково м.',
                        'Кунцевская м.',
                        'Авиамоторная м.', 'Тимирязевская м.', 'Столбово м.', 'Селигерская м.', 'Коньково м.',
                        'Бунинская аллея м.',
                        'Теплый стан м.', 'Ломоносовский проспект м.', 'Люблино м.', 'Электрозаводская м.', 'Перово м.',
                        'Бауманская м.', 'Автозаводская м.', 'Прокшино м.', 'Севастопольская м.', 'Новые Черемушки м.',
                        'Сходненская м.', 'Братиславская м.', 'Аннино м.', 'Ховрино м.', 'Преображенская площадь м.',
                        'Новокосино м.', 'Новопеределкино м.', 'Спартак м.', 'Беломорская м.', 'Тропарево м.',
                        'Академическая м.', 'Строгино м.', 'Выхино м.', 'Тульская м.', 'Говорово м.', 'Текстильщики м.',
                        'ЗИЛ м. (МЦК)', 'Домодедовская м.', 'Коломенская м.', 'Борисово м.', 'Волжская м.',
                        'Улица Скобелевская м.',
                        'Улица 1905 года м.', 'Андроновка м. (МЦК)', 'Нижегородская м. (МЦК)', 'Рассказовка м.',
                        'Отрадное м.',
                        'Окружная м.', 'Рязанский проспект м.', 'Свиблово м.', 'Солнцево м.', 'Сокол м.',
                        'Крымская м. (МЦК)', 'Выставочная м.',
                        'Румянцево м.', 'Ясенево м.', 'Смоленская м.', 'Новохохловская м. (МЦК)', 'Боровское шоссе м.',
                        'Шипиловская м.',
                        'Аэропорт м.', 'Щелковская м.', 'ВДНХ м.', 'Южная м.', 'Профсоюзная м.',
                        'Волгоградский проспект м.',
                        'Лесопарковая м.', 'Кропоткинская м.', 'Окская улица м.', 'Пролетарская м.', 'Стахановская м.',
                        'Озерная м.',
                        'Ольховая м.', 'Филатов луг м.', 'Новогиреево м.', 'Петровско-Разумовская м.', 'Некрасовка м.',
                        'Калужская м.',
                        'Площадь Гагарина м. (МЦК)', 'Лухмановская м.', 'Белорусская м.', 'Кантемировская м.',
                        'Бульвар Рокоссовского м. (МЦК)',
                        'Улица Горчакова м.', 'Красносельская м.', 'Бибирево м.', 'Фили м.', 'Владыкино м.',
                        'Алексеевская м.',
                        'Волоколамская м.', 'Новослободская м.', 'Угрешская м. (МЦК)', 'Курская м.',
                        'Лихоборы м. (МЦК)',
                        'Кузьминки м.', 'ЦСКА м.', 'Лермонтовский проспект м.', 'Алтуфьево м.', 'Динамо м.',
                        'Краснопресненская м.',
                        'Дмитровская м.', 'Соколиная Гора м. (МЦК)', 'Шоссе Энтузиастов м.', 'Семеновская м.',
                        'Чистые пруды м.',
                        'Красные ворота м.', 'Нахимовский проспект м.', 'Орехово м.', 'Технопарк м.', 'Бутырская м.',
                        'Студенческая м.', 'Щукинская м.', 'Жулебино м.', 'Чкаловская м.', 'Фонвизинская м.',
                        'Новоясеневская м.',
                        'Улица Старокачаловская м.', 'Речной вокзал м.', 'Митино м.', 'Юго-Восточная м.',
                        'Сокольники м.',
                        'Петровский парк м.', 'Пражская м.', 'Университет м.', 'Панфиловская м. (МЦК)',
                        'Автозаводская м. (МЦК)',
                        'Войковская м.', 'Беляево м.', 'Киевская м.', 'Черкизовская м.', 'Стрешнево м. (МЦК)',
                        'Тушинская м.',
                        'Тверская м.', 'Варшавская м.', 'Таганская м.', 'Китай-Город м.', 'Римская м.',
                        'Балтийская м. (МЦК)',
                        'Чертановская м.', 'Каховская м.', 'Бульвар Адмирала Ушакова м.', 'Полежаевская м.',
                        'Беговая м.',
                        'Дубровка м. (МЦК)', 'Парк культуры м.', 'Новокузнецкая м.', 'Парк Победы м.',
                        'Воробьевы Горы м.',
                        'Водный стадион м.', 'Победа станция', 'Достоевская м.', 'Кожуховская м.', 'Савеловская м.',
                        'Марксистская м.',
                        'Окружная м. (МЦК)', 'Алма-Атинская м.', 'Зорге м. (МЦК)', 'Менделеевская м.', 'Хорошевское м.',
                        'Нагорная м.','Крекшино станция','Крестьянская Застава м.','Комсомольская м.','Партизанская м.',
                        'Локомотив м. (МЦК)','Октябрьская м.','Савеловская метро','Дубровка м.','Измайлово м. (МЦК)',
                        'Мичуринец станция','Верхние Лихоборы м.','Каширская м.','Пушкинская м.','Филевский парк м.',
                        'Павелецкая м.',
                        'Кутузовская м.',
                        'Минская м.',
                        'Баррикадная м.',
                        'Библиотека им.Ленина м.',
                        'Зябликово м.',
                        'Аэропорт Внуково станция',
                        'Пионерская м.',
                        'Котельники м.',
                        'Деловой центр м.',
                        'Рассудово станция',
                        'Ленинский проспект м.',
                        'Рижская м.',
                        'Шелепиха м. (МЦК)',
                        'Сухаревская м.',
                        'Сретенский бульвар м.',
                        'Цветной бульвар м.',
                        'Трубная м.',
                        'Нагатинская м.',
                        'Третьяковская м.',
                        'Дачная станция',
                        'Проспект Мира м.',
                        'Охотный ряд м.',
                        'Боровицкая м.',
                        'Нижегородская улица м.',
                        'Международная м.',
                        'Весенняя станция',
                        'Багратионовская м.',
                        'Шоссе Энтузиастов м. (МЦК)',
                        'Полянка м.',
                        'Добрынинская м.',
                        'Лужники м. (МЦК)',
                        'Ботанический сад м. (МЦК)',
                        'Тургеневская м.',
                        'Щербинка станция',
                        'Бекасово-1 станция',
                        'Верхние Котлы м. (МЦК)',
                        'Апрелевка станция',
                        'Толстопальцево станция',
                        'Аэропорт (старая) станция',
                        'Силикатная станция',
                        'Лубянка м.',
                        'Кокошкино станция',
                        'Гривно станция', 'Косино м.', 'Подольск станция', 'Мякинино м.', 'Львовская станция',
                        'Внуково станция',
                        'Александровский Сад м.', 'Театральная м.', 'Кутузовская м. (МЦК)', 'Коммунарка м.',
                        'Битцевский парк м.',
                        'Колхозная станция', 'Чеховская м.', 'Алабушево станция', 'Площадь Революции м.',
                        'Кузнецкий мост м.'
                        ]

        #allparam1 = option + option1 + option2 + option3 + option4 + option5 + option6

        allparam2 = []
        allparam2.append(str(option))
        allparam2.append(str(option1))
        allparam2.append(str(option2))
        allparam2.append(str(option3))
        allparam2.append(str(option4))
        allparam2.append(str(option5))
        allparam2.append(str(option6))
        allparam2 = np.array(allparam2)



        roomsCountType = getRoomsCountCategory(allparam2, 30)
        metro = getMetro(allparam2, allStations1)
        metroType = getMetroType(allparam2)
        metroDistance = getMetroDistance(allparam2)
        floor, floors, isLastFloor, houseType = getHouseTypeAndFloor(allparam2)
        balcony = getBalcony(allparam2)
        wc = getWC(allparam2)
        area = getArea(allparam2)

        area = (area / 45)

        out = list(roomsCountType)
        out.append(float(metro))
        out.extend(metroType)
        out.extend(metroDistance)
        out.extend(floor)
        out.extend(floors)
        out.append(float(isLastFloor))
        out.extend(houseType)
        out.extend(balcony)
        out.extend(wc)
        out.append(area)
        out = [out]
        out = np.array(out)

        model = load_model('housemodel1.h5')
        pred = model.predict([out])  # Полуаем выход сети на проверочной выборке

        NewScaler1 = StandardScaler()
        NewScaler1.mean_ = np.array([9321217.94353961])
        NewScaler1.scale_ = np.array([22612982.87946659])
        UnscaledPred = NewScaler1.inverse_transform(np.array(pred).reshape(-1, 1))

        result11 = float(pred[0])
        result12 = float(UnscaledPred[0])

        #'Значение полученное с помощью линейной регрессии', result11
        'Нормируем значение, полученное с помощью нейронной сети'
        st.subheader(f"{result12:,} руб.")


'точечный график результатов предсказаний'
image11 = Image.open('/sysroot/home/user/Загрузки/PyProject/housereg3/venv/распределение_ошибки.png')
st.image(image11, caption='график ошибки')
#MAE, predUnscaled, yTrainUnscaled = mae(xTrainScaled[valMask], yTrainScaled[valMask])
#fig = plt.figure(figsize=(10,10))
#ax = fig.add_subplot(1, 1, 1)

# fig, axs = plt.subplots(1, 1, figsize = (30, 10))
#ax.scatter(predUnscaled, yTrainUnscaled)
#ax.set_xlabel('Реальные значения')
#ax.set_ylabel('Предсказания')
#ax.grid('on')
#ax.axis('equal')
#st.write(fig)


'график удаленности от центра'

image12 = Image.open('/sysroot/home/user/Загрузки/PyProject/housereg3/venv/график_удаленности_от_центра.png')
st.image(image12, caption='график удаленности станций от центра')

#center_metro = []
#ring_metro = []
#metro1_3 = []
#metro4_8 = []
#metro8_11 = []

#allparam2 = np.append(allparam2, result12)
#allparam2 = np.append(allparam2,[0,0,0,0,0,'примечание'])
#allparam2 = allparam2.reshape(1, 14)

#for i in range(data2.shape[0]):
#    if np.argmax(getMetroType(data2[i])) == 0:
#        center_metro.append(data2[i])
#    elif np.argmax(getMetroType(data2[i])) == 1:
#        ring_metro.append(data2[i])
#    elif np.argmax(getMetroType(data2[i])) == 2:
#        metro1_3.append(data2[i])
#    elif np.argmax(getMetroType(data2[i])) == 3:
#        metro4_8.append(data2[i])
#    elif np.argmax(getMetroType(data2[i])) == 4:
#        metro8_11.append(data2[i])

#scatter_fig = plt.figure(figsize=(10, 3))
#scatter_ax = scatter_fig.add_subplot(111)

#center_metro = pd.DataFrame(center_metro, columns=['Комнат', 'Метро / ЖД станции', 'От станции', 'Дом', 'Балкон', 'Санузел','Площадь', 'Цена, руб.', 'ГРМ', 'Бонус агенту', 'Дата','Кол-во дней в экспозиции', 'Источник', 'Примечание'])
#ring_metro = pd.DataFrame(ring_metro,columns=['Комнат', 'Метро / ЖД станции', 'От станции', 'Дом', 'Балкон', 'Санузел','Площадь','Цена, руб.', 'ГРМ', 'Бонус агенту', 'Дата','Кол-во дней в экспозиции', 'Источник', 'Примечание'])
#metro1_3 = pd.DataFrame(metro1_3,columns=['Комнат', 'Метро / ЖД станции', 'От станции', 'Дом', 'Балкон', 'Санузел','Площадь', 'Цена, руб.', 'ГРМ', 'Бонус агенту', 'Дата','Кол-во дней в экспозиции', 'Источник', 'Примечание'])
#metro4_8 = pd.DataFrame(metro4_8,columns=['Комнат', 'Метро / ЖД станции', 'От станции', 'Дом', 'Балкон', 'Санузел','Площадь', 'Цена, руб.', 'ГРМ', 'Бонус агенту', 'Дата','Кол-во дней в экспозиции', 'Источник', 'Примечание'])
        #predicted_value = pd.DataFrame(allparam2,columns=['Комнат', 'Метро / ЖД станции', 'От станции', 'Дом', 'Балкон', 'Санузел','Площадь', 'Цена, руб.', 'ГРМ', 'Бонус агенту', 'Дата','Кол-во дней в экспозиции', 'Источник', 'Примечание'])


        #metro8_11 = pd.DataFrame(metro8_11,columns=['Комнат', 'Метро / ЖД станции', 'От станции', 'Дом', 'Балкон', 'Санузел','Площадь', 'Цена, руб.', 'ГРМ', 'Бонус агенту', 'Дата','Кол-во дней в экспозиции', 'Источник', 'Примечание'])
#center_metro.plot.scatter(x='Метро / ЖД станции', y='Цена, руб.', s=20, c="tomato", alpha=0.6, ax=scatter_ax,label="Центральные")
#ring_metro.plot.scatter(x='Метро / ЖД станции', y='Цена, руб.', s=20, c="dodgerblue", alpha=0.6, ax=scatter_ax,label="Кольцевые")
#metro1_3.plot.scatter(x='Метро / ЖД станции', y='Цена, руб.', s=20, c="skyblue", alpha=0.6, ax=scatter_ax,label="В пределах 1-3 станций")
#metro4_8.plot.scatter(x='Метро / ЖД станции', y='Цена, руб.', s=20, c="yellowgreen", alpha=0.6, ax=scatter_ax,label="В пределах 4-8 станций")
#predicted_value.plot.scatter(x='Метро / ЖД станции', y='Цена, руб.', s=20, c="red", alpha=0.6, ax=scatter_ax,label="Предсказанное значение")
# metro8_11.plot.scatter(x='Метро / ЖД станции', y='Цена, руб.', s=120, c="hotpink", alpha=0.6, ax=scatter_ax, label= "За пределами 8 станций")

#scatter_ax.tick_params(axis='x', labelrotation=90, labelsize=4)
#scatter_ax.ticklabel_format(axis='y',scilimits=(0,30000000))
#st.write(scatter_fig)
