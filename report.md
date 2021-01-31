Краткий отчет

Мое решение состояло из не скольких этапов:

1) препроцессинг данных и генерация признаков 
   
Наиболее важная часть решения задания, так как от признаков зависит
   итоговый скор. Какие признаки я сделала:

весь код для генерации признаков находится здесь
python lib/code_for_learning/features.py  

2) выбор оптимальной модели:
   
   для табличных заданий всегда хорошо подходят градиентные бустинги, поэтому я использовала
   lighgbm. я сделала простой перебор гиперпараметров и выбрала наиболее важные на кросс-валидации


3) настройка кросс-валидации 
   
   Я использовала 5 fold кросс-валидацию на тренировочной выборке (на 800_000 из 900_
   объектов)
   100_000 отправились в мой собственный отложенный тест

   скрипт для кроссвалидации python lib/code_for_learning/cross_validation.py

   скор на кросс-валидации: ~ 0.93110 
   скор на отложенном тесте ~ 0.934
   
4) тестирование решения 
    каждый скрипт можно протестировать отдельно с помощью определенной команды
