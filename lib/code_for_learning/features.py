# coding: utf-8
import pandas as pd
import numpy as np
import re


class FeatureMaker:
    def __init__(self):
        self.is_fitted = False
        self.list_of_features = []

    def fit_and_transform_train(self, train):
        self.list_of_features = []

        train['datetime_submitted'] = pd.to_datetime(train['datetime_submitted'])
        train["description"] = train["description"].str.lower()

        train['price'] = train['price'].fillna(0)
        self.list_of_features.append('price')

        # subcategory_smoothed
        # делаем преобразование по каждой категории и создаем колонку соответствующую значению 
        # сглаживания по категории
        self.subcategory_smoothed_dict = dict(FeatureMaker.target_smoothing(train, 'subcategory'))
        train['subcategory_smoothed'] = train['subcategory'].map(self.subcategory_smoothed_dict)
        self.list_of_features.append('subcategory_smoothed')

        # category_smoothed
        self.category_smoothed_dict = dict(FeatureMaker.target_smoothing(train, 'category'))
        train['category_smoothed'] = train['category'].map(self.category_smoothed_dict)
        self.list_of_features.append('category_smoothed')

        # region_smoothed
        self.region_smoothed_dict = dict(FeatureMaker.target_smoothing(train, 'region'))
        train['region_smoothed'] = train['region'].map(self.region_smoothed_dict)
        self.list_of_features.append('region_smoothed')

        # city_smoothed
        self.city_smoothed_dict = dict(FeatureMaker.target_smoothing(train, 'city'))
        train['city_smoothed'] = train['city'].map(self.city_smoothed_dict)
        self.list_of_features.append('city_smoothed')

        train['month'] = train['datetime_submitted'].dt.month
        train['day'] = train['datetime_submitted'].dt.day
        train['hour'] = train['datetime_submitted'].dt.hour
        train['minute'] = train['datetime_submitted'].dt.minute
        train['dayofweek'] = train['datetime_submitted'].dt.dayofweek

        self.list_of_features.append('month')
        self.list_of_features.append('day')
        self.list_of_features.append('hour')
        self.list_of_features.append('minute')
        self.list_of_features.append('dayofweek')

        self.list_of_substrings = ['тел', 'ru', 'номер', 'звон', 'сайт', 'адрес', 'находит', 'связ', 'место',
                                   'вопрос', 'доставк', 'компан', 'писат', 'контакт', 'обращат', 'пишит', 'youtu',
                                   'ул.', 'фото', 'imei', '00', 'артикул', 'торг', 'работаем', 'уточн']

        self.create_find_substring_features_by_list(data=train,
                                                    list_of_substrings=self.list_of_substrings,
                                                    add_to_list_of_strings=True)

        train['find_number'] = train['description'].apply(lambda x: self.find_number(x)) * 1
        self.list_of_features.append('find_number')
        train['find_string_numbers'] = train['description'].apply(lambda x: self.find_string_numbers(x)) * 1
        self.list_of_features.append('find_string_numbers')

        train['find_email'] = train['description'].apply(lambda x: self.find_email(x))
        train['find_whats_app'] = train['description'].apply(lambda x: self.find_whats_app(x))
        train['find_telegram'] = train['description'].apply(lambda x: self.find_telegram(x))
        train['find_viber'] = train['description'].apply(lambda x: self.find_viber(x))
        train['find_vk'] = train['description'].apply(lambda x: self.find_vk(x))
        train['find_instagram'] = train['description'].apply(lambda x: self.find_instagram(x))

        self.list_of_features.append('find_email')
        self.list_of_features.append('find_whats_app')
        self.list_of_features.append('find_telegram')
        self.list_of_features.append('find_viber')
        self.list_of_features.append('find_vk')
        self.list_of_features.append('find_instagram')

        train['count_of_numbers'] = train['description'].apply(lambda x: self.count_of_numbers(x))
        train['count_of_letters'] = train['description'].apply(lambda x: self.count_of_letters(x))
        train['count_of_numbers_to_letters'] = train['description'].apply(lambda x: self.count_of_numbers_to_letters(x))

        self.list_of_features.append('count_of_numbers')
        self.list_of_features.append('count_of_letters')
        self.list_of_features.append('count_of_numbers_to_letters')

    def find_substring_in_description(self, data, substring, add_to_list_of_strings=True):
        new_feature_name = 'find_{}'.format(str(substring))
        data[new_feature_name] = data['description'].apply(lambda x: x.find(substring) >= 0) * 1
        if add_to_list_of_strings:
            self.list_of_features.append(new_feature_name)

    def create_find_substring_features_by_list(self, data, list_of_substrings, add_to_list_of_strings=True):
        for substring in list_of_substrings:
            self.find_substring_in_description(data=data, substring=substring,
                                               add_to_list_of_strings=add_to_list_of_strings)

    def transform_test(self, test):
        test['datetime_submitted'] = pd.to_datetime(test['datetime_submitted'])
        test["description"] = test["description"].str.lower()

        test['price'] = test['price'].fillna(0)

        test['subcategory_smoothed'] = test['subcategory'].map(self.subcategory_smoothed_dict)

        test['category_smoothed'] = test['category'].map(self.category_smoothed_dict)
        test['region_smoothed'] = test['region'].map(self.region_smoothed_dict)
        test['city_smoothed'] = test['city'].map(self.city_smoothed_dict)

        test['month'] = test['datetime_submitted'].dt.month
        test['day'] = test['datetime_submitted'].dt.day
        test['hour'] = test['datetime_submitted'].dt.hour
        test['minute'] = test['datetime_submitted'].dt.minute
        test['dayofweek'] = test['datetime_submitted'].dt.dayofweek

        self.create_find_substring_features_by_list(data=test,
                                                    list_of_substrings=self.list_of_substrings,
                                                    add_to_list_of_strings=False)

        test['find_number'] = test['description'].apply(lambda x: self.find_number(x)) * 1
        test['find_string_numbers'] = test['description'].apply(lambda x: self.find_string_numbers(x)) * 1

        test['find_email'] = test['description'].apply(lambda x: self.find_email(x))
        test['find_whats_app'] = test['description'].apply(lambda x: self.find_whats_app(x))
        test['find_telegram'] = test['description'].apply(lambda x: self.find_telegram(x))
        test['find_viber'] = test['description'].apply(lambda x: self.find_viber(x))
        test['find_vk'] = test['description'].apply(lambda x: self.find_vk(x))
        test['find_instagram'] = test['description'].apply(lambda x: self.find_instagram(x))

        test['count_of_numbers'] = test['description'].apply(lambda x: self.count_of_numbers(x))
        test['count_of_letters'] = test['description'].apply(lambda x: self.count_of_letters(x))
        test['count_of_numbers_to_letters'] = test['description'].apply(lambda x: self.count_of_numbers_to_letters(x))

    @staticmethod
    def target_smoothing(data, feature_name, target_name='is_bad', C=10):
        """
        сглаженная средняя целевой переменной
        https://habr.com/ru/company/yandex/blog/333440/
        """
        K = data.groupby(feature_name).size()
        mean_y = data.groupby(feature_name)[target_name].mean()
        global_mean_y = data[target_name].mean()
        return (mean_y * K + global_mean_y * C) / (K + C)

    @staticmethod
    def check_number(descr, number_position, len_number):
        has_number = False
        if number_position >= 0 and len(descr) > number_position + len_number:
            has_number = descr[number_position + len_number].isnumeric()
        return has_number

    @staticmethod
    def find_number(descr):
        """
        функция для поиска телефонного номера
        """
        phone = re.findall(r'(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)\s*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})',
                           descr)
        if len(phone) > 0:
            return True
        number_position = FeatureMaker.find_all(descr, '9')
        for pos in number_position:
            for j in range(9, 19):
                if FeatureMaker.check_number(descr, pos, j):
                    return True
        return False

    @staticmethod
    def find_email(descr):
        counter = 0
        mail_character = ['www', '@', 'ru', 'ру', 'точка', 'com', 'mail',
                          'яндекс' 'yandex', 'http', 'https', 'рф', 'маил', 'почта']
        for character in mail_character:
            if descr.find(character) != -1:
                counter += 1
        return counter

    @staticmethod
    def find_whats_app(descr):
        whats_app_character = ['ватс', 'what', 'вац', 'воц', 'вотс', 'wat', 'whатсaп']
        for character in whats_app_character:
            if descr.find(character) != -1:
                return True
        return False

    @staticmethod
    def find_telegram(descr):
        return ((descr.find('телег') != -1) or (descr.find('teleg') != -1)) * 1

    @staticmethod
    def find_viber(descr):
        return ((descr.find('viber') != -1) or (descr.find('вайбер') != -1) or (descr.find('вибер') != -1)) * 1

    @staticmethod
    def find_vk(descr):
        return ((descr.find('vk') != -1) or (descr.find('вконтакт') != -1) or (descr.find('vk.com') != -1))

    @staticmethod
    def find_instagram(descr):
        return ((descr.find('inst') != -1) or (descr.find('инст') != -1))

    @staticmethod
    def find_string_numbers(descr):
        str_nums = ['ноль', 'один', 'десять', ' сто ', 'два',
                    ' две', ' двенад', ' три', 'четыр', 'сорок', 'пять',
                    'шесть', 'семь', 'восем', 'девя']
        count = 0
        for num in str_nums:
            num_all = FeatureMaker.find_all(descr, num)
            for i in num_all:
                count += 1
        return count

    @staticmethod
    def find_all(a_str, sub):
        start = 0
        while True:
            start = a_str.find(sub, start)
            if start == -1: return
            yield start
            start += len(sub)

    @staticmethod
    def count_of_numbers(descr):
        return len(re.sub("[^0-9]", "", descr))

    @staticmethod
    def count_of_letters(descr):
        return len(descr)

    @staticmethod
    def count_of_numbers_to_letters(descr):
        return FeatureMaker.count_of_numbers(descr) / max(FeatureMaker.count_of_letters(descr), 1)


# testing FeatureMaker
# python lib/code_for_learning/features.py 
if __name__ == "__main__":
    train = pd.read_csv("/hiring-test-data/train_part.csv")
    test = pd.read_csv("/hiring-test-data/test.csv")

    feature_maker = FeatureMaker()
    feature_maker.fit_and_transform_train(train)

    print(feature_maker.list_of_features)

    feature_maker.transform_test(test)
