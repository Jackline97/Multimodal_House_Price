# Import packages and modules
import pandas as pd
import os
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
#from nltk.corpus import stopwords
import re
from num2words import num2words
from gensim.models import Word2Vec
import numpy as np 
from tensorflow import keras
from tensorflow.keras.models import load_model
#from sklearn.preprocessing import MinMaxScaler
import joblib
output_folder="models"

def preprocessing(description):
    df=pd.DataFrame({"description":[description]})
    #nltk.download('stopwords') 
    nltk.download('wordnet')

    ## convert to lower case
    df["description"]=df["description"].str.lower()

    ## Therefore, remove the 8 digits at the end.
    df["description"]=df["description"].str.replace("\(\d\d\d\d\d\d\d\d\)", "")

    ## Fix some abbreviations
    # replace Apostrophe ' with no string
    df["description"]=df["description"].str.replace('\'', "")

    df["description"]=df["description"].str.replace('.5 ', " and a half ")

    df["description"]=df["description"].str.replace(' c-vac ', " cvac ")

    df["description"]=df["description"].str.replace(' d/w', " dishwasher ")

    df["description"]=df["description"].str.replace(' w&d', " washer and dryer ")

    df["description"]=df["description"].str.replace(' d/w', " dryer ")

    df["description"]=df["description"].str.replace('w/o', " walk out ")

    df["description"]=df["description"].str.replace(r' [wW][/\s]r[.\s]', " washroom ")

    df["description"]=df["description"].str.replace(r' [wW]/', " with ")

    df["description"]=df["description"].str.replace(r' [wW]\&[dD][.\s]', " washer and dryer ")

    df["description"]=df["description"].str.replace(r'\ss/s\s', " ss ")

    df["description"]=df["description"].str.replace('bdrm', "bedroom")

    df["description"]=df["description"].str.replace('bsmt', "basement")

    df["description"]=df["description"].str.replace('+', " plus ")

    df["description"]=df["description"].str.replace('&', " and ")
    df["description"]=df["description"].str.replace('\$', " CAD ")
    df["description"]=df["description"].str.replace('don\'t', " do not ")
    df["description"]=df["description"].str.replace('doesn\'t', " does not ")
    df["description"]=df["description"].str.replace('can\'t', " cannot ")

    df["description"]=df["description"].str.replace('\'ll ', " will ")

    df['description']=df['description'].apply(lambda x: re.sub(r'(\d+),(\d+)', r'\1\2', x))


    ## Tokenization to remove the special characters

    tokeniser = RegexpTokenizer(r'\w+')
    df["description"]=df["description"].apply(lambda x: tokeniser.tokenize(x))

    ## detokenlization
    df["description"]=df["description"].apply(lambda x: " ".join(word for word in x))

    ## convert ordinal number to words, such as 1st--> first
    def replace_ordinal_numbers(text):
        re_results = re.findall('(\d+(st|nd|rd|th))', text)
        for enitre_result, suffix in re_results:
            num = int(enitre_result[:-2])
            text = text.replace(enitre_result, num2words(num, ordinal=True))
        return text
    
    df["description"]=df["description"].apply(lambda x: replace_ordinal_numbers(x))

    ## convert " w " or " W " to "with"
    df["description"]=df["description"].str.replace(r' [wW] ', " with ")

    ## remove single characters except "a" or number first
    df["description"]=df["description"].str.replace(r'\s[bcdefghijklmnopqrstuvwxyzBCDEFGHIJKLMNOPQRSTUVWXYZ]\s', " ")
    df["description"]=df["description"].str.replace(r'\s[bcdefghijklmnopqrstuvwxyzBCDEFGHIJKLMNOPQRSTUVWXYZ]\s', " ")
    df["description"]=df["description"].str.replace(r'\s[bcdefghijklmnopqrstuvwxyzBCDEFGHIJKLMNOPQRSTUVWXYZ]\s', " ")
    df["description"]=df["description"].str.replace(r'\s[bcdefghijklmnopqrstuvwxyzBCDEFGHIJKLMNOPQRSTUVWXYZ]$', " ")
    df["description"]=df["description"].str.replace(r'^[bcdefghijklmnopqrstuvwxyzBCDEFGHIJKLMNOPQRSTUVWXYZ]\s', " ")

    ## deal with abbreviation
    df["description"]=df["description"].str.replace(' 1st ', " first ")
    df["description"]=df["description"].str.replace(' 1st$', " first ")
    df["description"]=df["description"].str.replace('^1st ', " first ")

    df["description"]=df["description"].str.replace(' 2nd ', " second ")
    df["description"]=df["description"].str.replace(' 2nd$', " second ")
    df["description"]=df["description"].str.replace('^2nd ', " second ")

    df["description"]=df["description"].str.replace(' 3rd ', " third ")
    df["description"]=df["description"].str.replace(' 3rd$', " third ")
    df["description"]=df["description"].str.replace('^3rd ', " third ")


    df["description"]=df["description"].str.replace(' 4th ', " fourth ")
    df["description"]=df["description"].str.replace(' 4th$', " fourth ")
    df["description"]=df["description"].str.replace('^4th ', " fourth ")

    df["description"]=df["description"].str.replace(' 5th ', " fifth ")
    df["description"]=df["description"].str.replace(' 5th$', " fifth ")
    df["description"]=df["description"].str.replace('^5th ', " fifth ")

    df["description"]=df["description"].str.replace(' 6th ', " sixth ")
    df["description"]=df["description"].str.replace(' 6th$', " sixth ")
    df["description"]=df["description"].str.replace('^6th ', " sixth ")

    df["description"]=df["description"].str.replace(' 7th ', " seventh ")
    df["description"]=df["description"].str.replace(' 7th$', " seventh ")
    df["description"]=df["description"].str.replace('^7th ', " seventh ")

    df["description"]=df["description"].str.replace(' 8th ', " eighth ")
    df["description"]=df["description"].str.replace(' 8th$', " eighth ")
    df["description"]=df["description"].str.replace('^8th ', " eighth ")

    df["description"]=df["description"].str.replace(' 9th ', " ninth ")
    df["description"]=df["description"].str.replace(' 9th$', " ninth ")
    df["description"]=df["description"].str.replace('^9th ', " ninth ")


    ## seperate number and alphabetical characters
    df["description"]=df["description"].str.replace('(\d+(\.\d+)?)', r' \1 ')

    ## sep usually means "seperate"
    df["description"]=df["description"].str.replace(' sep ', " separate ")

    ## hwy means "highway"
    df["description"]=df["description"].str.replace(' hwy ', " highway ")
    df["description"]=df["description"].str.replace(' hwy$', " highway ")
    df["description"]=df["description"].str.replace('^hwy ', " highway ")

    ## bdr, bedrm means bedroom
    ## washrm means washroom
    ##  rm means room 

    df["description"]=df["description"].str.replace(' bdr ', " bedroom ")
    df["description"]=df["description"].str.replace(' bdr$', " bedroom ")
    df["description"]=df["description"].str.replace('^bdr ', " bedroom ")

    df["description"]=df["description"].str.replace('bedrm', "bedroom")
    df["description"]=df["description"].str.replace('washrm', "washroom")
    df["description"]=df["description"].str.replace('bathrm', "bathroom")

    df["description"]=df["description"].str.replace(' rm ', " room ")

    ## convert square feet to sqft

    df["description"]=df["description"].str.replace(' sq ft ', " sqft ")
    df["description"]=df["description"].str.replace('sqft', " sqft ")
    df["description"]=df["description"].str.replace(' sqf ', " sqft ")
    df["description"]=df["description"].str.replace(' sf ', " sqft ")


    df["description"]=df["description"].str.replace('square ft', " sqft ")
    df["description"]=df["description"].str.replace('square feet', " sqft ")
    df["description"]=df["description"].str.replace('square foot', " sqft ")
    df["description"]=df["description"].str.replace('sq feet', " sqft ")
    df["description"]=df["description"].str.replace('sq foot', " sqft ")

    df["description"]=df["description"].str.replace(' sqft ', " square feet ")


    ## convert abbreviation and mis-spelled word to the correct form

    df["description"]=df["description"].str.replace('hardwd', "hardwood")

    df["description"]=df["description"].str.replace('addtl', "additional")
    df["description"]=df["description"].str.replace(' lvl ', " level ")
    df["description"]=df["description"].str.replace(' lvl$', " level ")
    df["description"]=df["description"].str.replace('^lvl ', " level ")

    df["description"]=df["description"].str.replace(' lwr ', " lower ")
    df["description"]=df["description"].str.replace('^lwr ', " lower ")
    df["description"]=df["description"].str.replace(' lwr$', " lower ")

    df["description"]=df["description"].str.replace(' fl ', " floor ")

    df["description"]=df["description"].str.replace(' flr ', " floor ")
    df["description"]=df["description"].str.replace(' flrs ', " floors ")
    df["description"]=df["description"].str.replace(' flring ', " flooring ")




    df["description"]=df["description"].str.replace(' reno ', " renovated ")

    df["description"]=df["description"].str.replace(' entr ', " entrance ")
    df["description"]=df["description"].str.replace(' ent ', " entrance ")

    df["description"]=df["description"].str.replace(' balciony ', " balcony ")

    df["description"]=df["description"].str.replace(' semidetached ', " semi detached ")



    df["description"]=df["description"].str.replace(' incl ', " include ")
    df["description"]=df["description"].str.replace(' excl ', " exclude ")
    df["description"]=df["description"].str.replace(' exl ', " exclude ")

    df["description"]=df["description"].str.replace(' pcs ', " pieces ")
    df["description"]=df["description"].str.replace(' pc ', " piece ")

    df["description"]=df["description"].str.replace('dinning', "dining")

    df["description"]=df["description"].str.replace(' stn ', " station ")

    df["description"]=df["description"].str.replace(' dw ', " dishwasher ")

    df["description"]=df["description"].str.replace(' ctr ', " center ")
    df["description"]=df["description"].str.replace(' centre ', " center ")

    df["description"]=df["description"].str.replace('rec room', "recreation room")

    df["description"]=df["description"].str.replace(' cvac ', " central vacuum ")
    df["description"]=df["description"].str.replace(' central vac ', " central vacuum ")


    df["description"]=df["description"].str.replace(' st ', " street ")

    df["description"]=df["description"].str.replace('bk yard', " back yard ")

    df["description"]=df["description"].str.replace(' backyard ', " back yard ")
    df["description"]=df["description"].str.replace(' ac ', " air conditioner ")

    df["description"]=df["description"].str.replace(' da ', " the ")

    df["description"]=df["description"].str.replace(' kit ', " kitchen ")

    df["description"]=df["description"].str.replace(' kitch ', " kitchen ")

    df["description"]=df["description"].str.replace(' u ', " you ")
    df["description"]=df["description"].str.replace(' ur ', " your ")
    df["description"]=df["description"].str.replace(' urself ', " yourself ")

    df["description"]=df["description"].str.replace(' what 2 do ', " what to do ")

    df["description"]=df["description"].str.replace(' dont ', " do not ")

    df["description"]=df["description"].str.replace(' cul de sac ', " culdesac ")

    df["description"]=df["description"].str.replace(' through out ', " throughout ")
    df["description"]=df["description"].str.replace(' thru out ', " throughout ")

    df["description"]=df["description"].str.replace(' hr ', " hour ")
    df["description"]=df["description"].str.replace(' bld ', " boulevard ")
    df["description"]=df["description"].str.replace(' remks ', " remarks ")

    df["description"]=df["description"].str.replace(' fire place ', " fireplace ")

    df["description"]=df["description"].str.replace(' ss ', " stainless steel ")

    df["description"]=df["description"].str.replace(' hwt ', " hot water tank ")
    df["description"]=df["description"].str.replace(' hwt$', " hot water tank ")

    df["description"]=df["description"].str.replace(' hwts ', " hot water tank ")
    df["description"]=df["description"].str.replace(' hwts$', " hot water tank ")


    df["description"]=df["description"].str.replace(' incl ', " include ")
    df["description"]=df["description"].str.replace(' incl$', " include ")


    df["description"]=df["description"].str.replace(' incls ', " include ")


    df["description"]=df["description"].str.replace(' inc ', " include ")

    df["description"]=df["description"].str.replace(' excl ', " exclude ")
    df["description"]=df["description"].str.replace(' exl ', " exclude ")
    df["description"]=df["description"].str.replace(' excls ', " exclude ")



    df["description"]=df["description"].str.replace(' pcs ', " pieces ")
    df["description"]=df["description"].str.replace(' pc ', " piece ")

    df["description"]=df["description"].str.replace('dinning', "dining")

    df["description"]=df["description"].str.replace(' stn ', " station ")

    df["description"]=df["description"].str.replace(' dw ', " dishwasher ")

    df["description"]=df["description"].str.replace(' ctr ', " center ")
    df["description"]=df["description"].str.replace(' centre ', " center ")

    df["description"]=df["description"].str.replace('rec room', "recreation room")

    df["description"]=df["description"].str.replace(' cvac ', " central vac ")


    df["description"]=df["description"].str.replace(' st ', " street ")

    df["description"]=df["description"].str.replace('bk yard', " back yard ")

    df["description"]=df["description"].str.replace(' back yard ', " backyard ")
    df["description"]=df["description"].str.replace(' ac ', " air conditioner ")

    df["description"]=df["description"].str.replace(' da ', " the ")

    df["description"]=df["description"].str.replace(' kit ', " kitchen ")

    df["description"]=df["description"].str.replace(' kitch ', " kitchen ")

    df["description"]=df["description"].str.replace(' kitn ', " kitchen ")



    df["description"]=df["description"].str.replace(' u ', " you ")
    df["description"]=df["description"].str.replace(' ur ', " your ")
    df["description"]=df["description"].str.replace(' urself ', " yourself ")

    df["description"]=df["description"].str.replace(' what 2 do ', " what to do ")

    df["description"]=df["description"].str.replace(' dont ', " do not ")
    df["description"]=df["description"].str.replace(' don miss', " do not miss")


    df["description"]=df["description"].str.replace(' cul de sac ', " culdesac ")

    df["description"]=df["description"].str.replace(' through out ', " throughout ")
    df["description"]=df["description"].str.replace(' thru out ', " throughout ")
    df["description"]=df["description"].str.replace(' thruout ', " throughout ")
    df["description"]=df["description"].str.replace(' thr out ', " throughout ")


    df["description"]=df["description"].str.replace(' thru ', " through ")



    df["description"]=df["description"].str.replace(' hr ', " hour ")
    df["description"]=df["description"].str.replace(' hrs ', " hours ")


    df["description"]=df["description"].str.replace(' bld ', " boulevard ")
    df["description"]=df["description"].str.replace(' remks ', " remarks ")

    df["description"]=df["description"].str.replace(' fire place ', " fireplace ")

    df["description"]=df["description"].str.replace(' ss ', " stainless steel ")

    df["description"]=df["description"].str.replace(' tvs ', " television ")
    df["description"]=df["description"].str.replace(' tv ', " television ")


    df["description"]=df["description"].str.replace(' br ', " bedroom ")
    df["description"]=df["description"].str.replace(' br$', " bedroom ")

    df["description"]=df["description"].str.replace(' brm ', " bedroom ")
    df["description"]=df["description"].str.replace(' brs ', " bedrooms ")

    df["description"]=df["description"].str.replace(' pre emptive ', " preemptive ")
    df["description"]=df["description"].str.replace(' pre ', " previously ")


    df["description"]=df["description"].str.replace(' eng hardwood ', " engineered hardwood ")
    df["description"]=df["description"].str.replace(' bksplsh ', " backsplash ")

    df["description"]=df["description"].str.replace(' bathflrs ', " bath floors ")

    df["description"]=df["description"].str.replace(' grnd ', " ground ")

    df["description"]=df["description"].str.replace(' fl washer ', " front load ")

    df["description"]=df["description"].str.replace(' gdo ', " garage door opener ")
    df["description"]=df["description"].str.replace(' gdo$', " garage door opener ")

    df["description"]=df["description"].str.replace(' cac ', " central air conditioner ")

    df["description"]=df["description"].str.replace(' quite street ', " quiet street ")

    df["description"]=df["description"].str.replace(' dist ', " distance ")

    df["description"]=df["description"].str.replace(' abv ', " above ")

    df["description"]=df["description"].str.replace(' bkfst ', " breakfast ")

    df["description"]=df["description"].str.replace(' lrg ', " large ")

    df["description"]=df["description"].str.replace(' hdwd ', " hardwood ")
    df["description"]=df["description"].str.replace(' hrdwd ', " hardwood ")


    df["description"]=df["description"].str.replace(' cntp ', " countertop ")

    df["description"]=df["description"].str.replace(' rntl ', " rental ")
    df["description"]=df["description"].str.replace(' rntl$', " rental ")


    df["description"]=df["description"].str.replace(' bbq ', " barbeque ")
    df["description"]=df["description"].str.replace(' bbq$', " barbeque ")

    df["description"]=df["description"].str.replace(' pls ', " please ")
    df["description"]=df["description"].str.replace(' pls$', " please ")

    df["description"]=df["description"].str.replace(' apt$', " apartment ")
    df["description"]=df["description"].str.replace(' apt ', " apartment ")
    df["description"]=df["description"].str.replace('^apt ', " apartment ")

    df["description"]=df["description"].str.replace(' lr ', " living room ")
    df["description"]=df["description"].str.replace(' dr ', " dining room ")

    df["description"]=df["description"].str.replace(' fp ', " fireplace ")

    df["description"]=df["description"].str.replace(' dble ', " double ")
    df["description"]=df["description"].str.replace(' dbl ', " double ")



    df["description"]=df["description"].str.replace(' gar ', " garage ")

    df["description"]=df["description"].str.replace(' approx ', " approximately ")
    df["description"]=df["description"].str.replace(' appr ', " approximately ")
    df["description"]=df["description"].str.replace('^appr ', " approximately ")


    df["description"]=df["description"].str.replace(' fab ', " fabulous ")
    df["description"]=df["description"].str.replace('^fab ', " fabulous ")
    df["description"]=df["description"].str.replace(' fab$', " fabulous ")


    df["description"]=df["description"].str.replace(' bed ', " bedroom ")

    df["description"]=df["description"].str.replace(' bath ', " bathroom ")

    df["description"]=df["description"].str.replace(' potlights ', " pot lights ")

    df["description"]=df["description"].str.replace(' ave ', " avenue ")

    df["description"]=df["description"].str.replace(' atn ', " attention ")

    df["description"]=df["description"].str.replace(' attn ', " attention ")



    df["description"]=df["description"].str.replace(' cre ', " crescent ")

    df["description"]=df["description"].str.replace(' cres ', " crescent ")

    df["description"]=df["description"].str.replace(' prem ', " premium ")

    df["description"]=df["description"].str.replace(' cntr ', " center ")

    df["description"]=df["description"].str.replace(' dvp ', " don valley parkway ")

    df["description"]=df["description"].str.replace(' lgl ', " legal ")

    df["description"]=df["description"].str.replace(' desc ', " description ")

    df["description"]=df["description"].str.replace(' twp ', " township ")

    #df["description"]=df["description"].str.replace(' ttc ', " toronto transit commission ")

    df["description"]=df["description"].str.replace(' maint ', " maintenance ")

    df["description"]=df["description"].str.replace(' incd ', " included ")
    df["description"]=df["description"].str.replace(' incd$', " included ")
    df["description"]=df["description"].str.replace(' incld ', " include ")



    df["description"]=df["description"].str.replace(' avbl ', " available ")
    df["description"]=df["description"].str.replace(' avbl$', " available ")


    df["description"]=df["description"].str.replace(' ft ', " feet ")
    df["description"]=df["description"].str.replace(' ft$', " feet ")
    df["description"]=df["description"].str.replace('^ft ', " feet ")

    df["description"]=df["description"].str.replace(' equiment ', " equipment ")
    df["description"]=df["description"].str.replace('^equiment ', " equipment ")
    df["description"]=df["description"].str.replace(' equiment$', " equipment ")

    df["description"]=df["description"].str.replace(' min ', " minute ")
    df["description"]=df["description"].str.replace(' mins ', " minute ")

    df["description"]=df["description"].str.replace(' uft ', " university of toronto ")
    df["description"]=df["description"].str.replace(' uoft ', " university of toronto ")

    df["description"]=df["description"].str.replace(' hm ', " home ")

    df["description"]=df["description"].str.replace(' mnth ', " month ")

    df["description"]=df["description"].str.replace(' apprx ', " approximately ")

    df["description"]=df["description"].str.replace(' sq ', " square ")
    df["description"]=df["description"].str.replace('^sq ', " square ")
    df["description"]=df["description"].str.replace(' sq$', " square ")

    df["description"]=df["description"].str.replace(' appx ', " approximate ")

    df["description"]=df["description"].str.replace(' you re ', " you are ")

    df["description"]=df["description"].str.replace(' ciity  ', " city ")

    df["description"]=df["description"].str.replace(' xcellent ', " excellent ")
    df["description"]=df["description"].str.replace('^xcellent ', " excellent ")
    df["description"]=df["description"].str.replace(' xcellent$', " excellent ")

    df["description"]=df["description"].str.replace(' mt pleasant ', " mount pleasant ")
    df["description"]=df["description"].str.replace('^mt pleasant ', " mount pleasant ")
    df["description"]=df["description"].str.replace(' mt pleasant$', " mount pleasant ")

    df["description"]=df["description"].str.replace(' liv ', " living ")

    df["description"]=df["description"].str.replace(' din ', " dining ")

    df["description"]=df["description"].str.replace(' ameneties ', " amenities ")

    df["description"]=df["description"].str.replace(' frt ', " front ")

    df["description"]=df["description"].str.replace(' centres ', " center ")

    df["description"]=df["description"].str.replace(' rec ', " recreation ")

    df["description"]=df["description"].str.replace(' crt ', " crescent ")

    df["description"]=df["description"].str.replace(' lg ', " large ")

    df["description"]=df["description"].str.replace(' walkin ', " walking ")

    df["description"]=df["description"].str.replace(' fin ', " finished ")

    df["description"]=df["description"].str.replace(' wdws ', " windows ")

    df["description"]=df["description"].str.replace(' wd ', " wood ")

    df["description"]=df["description"].str.replace(' egdo ', " electric garage door ")

    df["description"]=df["description"].str.replace(' hwys ', " highway ")
    df["description"]=df["description"].str.replace(' hwys$', " highway ")

    df["description"]=df["description"].str.replace(' fam ', " family ")

    df["description"]=df["description"].str.replace(' comm ', " community ")

    df["description"]=df["description"].str.replace(' prefect ', " perfect ")

    df["description"]=df["description"].str.replace(' washer dyer ', " washer dryer ")
    df["description"]=df["description"].str.replace(' washer and dyer ', " washer and dryer ")

    df["description"]=df["description"].str.replace(' det house ', " detached house ")

    df["description"]=df["description"].str.replace(' roughin ', " rough in ")

    df["description"]=df["description"].str.replace(' walkout ', " walk out ")


    df["description"]=df["description"].str.replace(' mldg ', " molding ")
    df["description"]=df["description"].str.replace(' mldgs ', " molding ")

    df["description"]=df["description"].str.replace(' prof ', " professional ")

    df["description"]=df["description"].str.replace(' carbon mon ', " carbon monoxide ")

    df["description"]=df["description"].str.replace(' chdlr ', " chandelier ")

    df["description"]=df["description"].str.replace(' condationer ', " conditioner ")

    df["description"]=df["description"].str.replace(' ext ', " extras ")

    df["description"]=df["description"].str.replace(' en suite ', " ensuite ")

    df["description"]=df["description"].str.replace(' air condition ', " air conditioner ")

    df["description"]=df["description"].str.replace(' air conditioning ', " air conditioner ")

    df["description"]=df["description"].str.replace(' refrigerator ', " fridge ")

    df["description"]=df["description"].str.replace(' tlc ', " tender loving care ")

    df["description"]=df["description"].str.replace(' nw facing ', " northwest facing ")
    df["description"]=df["description"].str.replace(' nw ', " northwest ")
    df["description"]=df["description"].str.replace(' sw ', " southwest ")


    df["description"]=df["description"].str.replace(' ffice ', " office ")

    df["description"]=df["description"].str.replace(' ens bathroom ', " ensuite bathroom ")

    df["description"]=df["description"].str.replace(' prkg ', " parking ")
    df["description"]=df["description"].str.replace(' prkg$', " parking ")

    df["description"]=df["description"].str.replace(' pking ', " parking ")
    df["description"]=df["description"].str.replace(' pking$', " parking ")



    df["description"]=df["description"].str.replace(' hdwr ', " hardware ")

    df["description"]=df["description"].str.replace(' est ', " estimated ")

    df["description"]=df["description"].str.replace(' estm ', " estimated ")


    df["description"]=df["description"].str.replace(' mn ', " main ")

    df["description"]=df["description"].str.replace(' linc ', " lincoln alexander parkway ")

    df["description"]=df["description"].str.replace(' won be ', " will not ")

    df["description"]=df["description"].str.replace(' blt in ', " built in ")

    df["description"]=df["description"].str.replace(' porcelin ', " porcelain ")

    df["description"]=df["description"].str.replace(' oc transpo ', " octranspo ")
    df["description"]=df["description"].str.replace(' oc transpo$', " octranspo ")


    df["description"]=df["description"].str.replace(' yr ', " year ")
    df["description"]=df["description"].str.replace('^yr ', " year ")
    df["description"]=df["description"].str.replace(' yr$', " year ")

    df["description"]=df["description"].str.replace(' yrs ', " years ")
    df["description"]=df["description"].str.replace('^yrs ', " years ")
    df["description"]=df["description"].str.replace(' yrs$', " years ")

    df["description"]=df["description"].str.replace(' detach home ', " detached home ")

    df["description"]=df["description"].str.replace(' mstr ', " master ")

    df["description"]=df["description"].str.replace(' mbr ', " master bedroom ")
    df["description"]=df["description"].str.replace(' mbr$', " master bedroom ")

    df["description"]=df["description"].str.replace(' utm ', " university of toronto mississauga ")

    df["description"]=df["description"].str.replace(' theater ', " theatre ")

    df["description"]=df["description"].str.replace(' nannysuite ', " nanny suite ")

    df["description"]=df["description"].str.replace(' neighborhood ', " neighbourhood ")

    df["description"]=df["description"].str.replace(' specious ', " spacious ")

    df["description"]=df["description"].str.replace(' roof top ', " rooftop ")

    df["description"]=df["description"].str.replace(' insur ', " insurance ")

    df["description"]=df["description"].str.replace(' buildin ', " building ")
    df["description"]=df["description"].str.replace('^buildin ', " building ")
    df["description"]=df["description"].str.replace(' buildin$', " building ")


    df["description"]=df["description"].str.replace(' bldg ', " building ")
    df["description"]=df["description"].str.replace('^bldg ', " building ")
    df["description"]=df["description"].str.replace(' bldg$', " building ")


    df["description"]=df["description"].str.replace(' bsmnt ', " basement ")

    df["description"]=df["description"].str.replace(' coin op ', " coin operated ")

    df["description"]=df["description"].str.replace(' turn key ', " turnkey ")

    df["description"]=df["description"].str.replace(' qew ', " queen elizabeth way ")
    df["description"]=df["description"].str.replace('^qew ', " queen elizabeth way ")
    df["description"]=df["description"].str.replace(' qew$', " queen elizabeth way ")


    df["description"]=df["description"].str.replace(' larage ', " large ")

    df["description"]=df["description"].str.replace(' lge ', " large ")
    df["description"]=df["description"].str.replace('^lge ', " large ")
    df["description"]=df["description"].str.replace(' lge$', " large ")



    df["description"]=df["description"].str.replace(' rangehood', " range hood ")

    df["description"]=df["description"].str.replace('breath taking', "breathtaking")

    df["description"]=df["description"].str.replace('gardnier', "gardiner")

    df["description"]=df["description"].str.replace('over the range microwave', "otr microwave")

    df["description"]=df["description"].str.replace('over range microwave', "otr microwave")

    df["description"]=df["description"].str.replace(' equiped ', " equipped ")

    df["description"]=df["description"].str.replace('water front', "waterfront")

    df["description"]=df["description"].str.replace(' pen concept ', " open concept ")

    df["description"]=df["description"].str.replace(' re modeled ', " remodeled ")

    df["description"]=df["description"].str.replace(' hottub ', " hot hub ")

    df["description"]=df["description"].str.replace(' carwash ', " car wash ")
    df["description"]=df["description"].str.replace(' carwash$', " car wash ")

    df["description"]=df["description"].str.replace(' back splash ', " backsplash ")
    df["description"]=df["description"].str.replace('^back splash ', " backsplash ")
    df["description"]=df["description"].str.replace(' back splash$', " backsplash ")

    df["description"]=df["description"].str.replace(' pce ', " piece ")
    df["description"]=df["description"].str.replace(' pice ', " piece ")


    df["description"]=df["description"].str.replace(' appl ', " appliance ")
    df["description"]=df["description"].str.replace(' app ', " appliance ")
    df["description"]=df["description"].str.replace(' appls ', " appliances ")


    df["description"]=df["description"].str.replace(' wsher ', " washer ")

    df["description"]=df["description"].str.replace(' attachd ', " attached ")
    df["description"]=df["description"].str.replace(' attachd$', " attached ")
    df["description"]=df["description"].str.replace('^attachd ', " attached ")

    df["description"]=df["description"].str.replace(' soughtafter ', " sought after ")

    df["description"]=df["description"].str.replace(' bthrms ', " bathrooms ")
    df["description"]=df["description"].str.replace(' bthrms$', " bathrooms ")
    df["description"]=df["description"].str.replace('^bthrms ', " bathrooms ")

    df["description"]=df["description"].str.replace(' bthrm ', " bathroom ")
    df["description"]=df["description"].str.replace(' bthrm$', " bathroom ")
    df["description"]=df["description"].str.replace('^bthrm ', " bathroom ")

    df["description"]=df["description"].str.replace(' you ve ', " you have ")
    df["description"]=df["description"].str.replace('^you ve ', " you have ")

    df["description"]=df["description"].str.replace(' strge ', " storage ")
    df["description"]=df["description"].str.replace(' strge$', " storage ")
    df["description"]=df["description"].str.replace('^strge ', " storage ")

    df["description"]=df["description"].str.replace(' lyr ', " layer ")
    df["description"]=df["description"].str.replace(' lyr$', " layer ")
    df["description"]=df["description"].str.replace('^lyr ', " layer ")

    df["description"]=df["description"].str.replace(' dish washer', " dishwasher ")

    df["description"]=df["description"].str.replace(' wr ', " washroom ")
    df["description"]=df["description"].str.replace(' wr$', " washroom ")

    df["description"]=df["description"].str.replace(' quite crescent ', " quiet crescent ")
    df["description"]=df["description"].str.replace('^quite crescent ', " quiet crescent ")
    df["description"]=df["description"].str.replace(' quite crescent$', " quiet crescent ")


    df["description"]=df["description"].str.replace(' side walk ', " sidewalk ")
    df["description"]=df["description"].str.replace('^side walk ', " sidewalk ")
    df["description"]=df["description"].str.replace(' side walk$', " sidewalk ")

    df["description"]=df["description"].str.replace(' mordern ', " modern ")
    df["description"]=df["description"].str.replace(' mordern$', " modern ")
    df["description"]=df["description"].str.replace('^mordern ', " modern ")

    df["description"]=df["description"].str.replace(' neighbors ', " neighbours ")
    df["description"]=df["description"].str.replace('^neighbors ', " neighbours ")
    df["description"]=df["description"].str.replace(' neighbors$', " neighbours ")

    df["description"]=df["description"].str.replace(' neighbor ', " neighbour ")
    df["description"]=df["description"].str.replace('^neighbor ', " neighbour ")
    df["description"]=df["description"].str.replace(' neighbor$', " neighbour ")

    df["description"]=df["description"].str.replace(' floorplan ', " floor plan ")

    df["description"]=df["description"].str.replace(' re development ', " redevelopment ")

    df["description"]=df["description"].str.replace(' ev ', " electric vehicle ")

    df["description"]=df["description"].str.replace(' super market ', " supermarket ")
    df["description"]=df["description"].str.replace(' super market$', " supermarket ")
    df["description"]=df["description"].str.replace('^super market ', " supermarket ")

    df["description"]=df["description"].str.replace(' etobicoke ci ', " etobicoke collegiate institute ")

    df["description"]=df["description"].str.replace(' insp ', " inspection ")
    df["description"]=df["description"].str.replace('^insp ', " inspection ")
    df["description"]=df["description"].str.replace(' insp$', " inspection ")

    df["description"]=df["description"].str.replace(' exec ', " executive ")
    df["description"]=df["description"].str.replace('^exec ', " executive ")
    df["description"]=df["description"].str.replace(' exec$', " executive ")

    df["description"]=df["description"].str.replace(' rms ', " rooms ")
    df["description"]=df["description"].str.replace('^rms ', " rooms ")
    df["description"]=df["description"].str.replace(' rms$', " rooms ")


    df["description"]=df["description"].str.replace(' hw ', " hardwood floor ")
    df["description"]=df["description"].str.replace('^hw ', " hardwood floor ")
    df["description"]=df["description"].str.replace(' hw$', " hardwood floor ")

    df["description"]=df["description"].str.replace(' hwf ', " hardwood floor ")
    df["description"]=df["description"].str.replace('^hwf ', " hardwood floor ")
    df["description"]=df["description"].str.replace(' hwf$', " hardwood floor ")



    df["description"]=df["description"].str.replace(' gb ', " gas burner ")
    df["description"]=df["description"].str.replace('^gb ', " gas burner ")
    df["description"]=df["description"].str.replace(' gb$', " gas burner ")

    df["description"]=df["description"].str.replace(' lph ', " lower penthouse ")
    df["description"]=df["description"].str.replace('^lph ', " lower penthouse ")
    df["description"]=df["description"].str.replace(' lph$', " lower penthouse ")

    df["description"]=df["description"].str.replace(' locaction ', " locaction ")
    df["description"]=df["description"].str.replace('^locaction ', " locaction ")
    df["description"]=df["description"].str.replace(' locaction$', " locaction ")

    df["description"]=df["description"].str.replace(' mic ', " microwave ")
    df["description"]=df["description"].str.replace(' mic$', " microwave ")

    df["description"]=df["description"].str.replace(' micro ', " microwave ")
    df["description"]=df["description"].str.replace(' micro$', " microwave ")



    df["description"]=df["description"].str.replace(' lckr ', " locker ")
    df["description"]=df["description"].str.replace(' lckr$', " locker ")
    df["description"]=df["description"].str.replace('^lckr ', " locker ")

    df["description"]=df["description"].str.replace(' counter tops ', " countertops ")
    df["description"]=df["description"].str.replace(' counter tops$', " countertops ")
    df["description"]=df["description"].str.replace('^counter tops ', " countertops ")

    df["description"]=df["description"].str.replace(' counter top ', " countertop ")
    df["description"]=df["description"].str.replace(' counter top$', " countertop ")
    df["description"]=df["description"].str.replace('^counter top ', " countertop ")

    df["description"]=df["description"].str.replace(' nac ', " national art center ")
    df["description"]=df["description"].str.replace(' nac$', " national art center ")
    df["description"]=df["description"].str.replace('^nac ', " national art center ")


    df["description"]=df["description"].str.replace(' south west ', " southwest ")

    df["description"]=df["description"].str.replace(' over sized ', " oversized ")

    df["description"]=df["description"].str.replace(' exisiting ', " existing ")

    df["description"]=df["description"].str.replace(' orig ', " original ")

    df["description"]=df["description"].str.replace(' spc ', " space ")
    df["description"]=df["description"].str.replace(' spc$', " space ")

    df["description"]=df["description"].str.replace(' fr ', " family room ")
    df["description"]=df["description"].str.replace(' fr$', " family room ")

    df["description"]=df["description"].str.replace(' stv ', " stove ")
    df["description"]=df["description"].str.replace(' stv$', " stove ")

    df["description"]=df["description"].str.replace(' wndw ', " window ")
    df["description"]=df["description"].str.replace('^wndw ', " window ")
    df["description"]=df["description"].str.replace(' wndw$', " window ")

    df["description"]=df["description"].str.replace(' crt ', " court ")
    df["description"]=df["description"].str.replace('^crt ', " court ")
    df["description"]=df["description"].str.replace(' crt$', " court ")

    df["description"]=df["description"].str.replace(' crts ', " courts ")
    df["description"]=df["description"].str.replace('^crts ', " courts ")
    df["description"]=df["description"].str.replace(' crts$', " courts ")

    df["description"]=df["description"].str.replace(' hobbyistes ', " hobbyistes ")
    df["description"]=df["description"].str.replace('^hobbyistes ', " hobbyistes ")
    df["description"]=df["description"].str.replace(' hobbyistes$', " hobbyistes ")

    df["description"]=df["description"].str.replace(' smarthome ', " smart home ")
    df["description"]=df["description"].str.replace('^smarthome ', " smart home ")
    df["description"]=df["description"].str.replace(' smarthome$', " smart home ")

    df["description"]=df["description"].str.replace(' cook top ', " cooktop ")
    df["description"]=df["description"].str.replace(' cook top$', " cooktop ")

    df["description"]=df["description"].str.replace(' bath ', " bathroom ")
    df["description"]=df["description"].str.replace('^bath ', " bathroom ")
    df["description"]=df["description"].str.replace(' bath$', " bathroom ")

    df["description"]=df["description"].str.replace(' baths ', " bathrooms ")
    df["description"]=df["description"].str.replace('^baths ', " bathrooms ")
    df["description"]=df["description"].str.replace(' baths$', " bathrooms ")

    df["description"]=df["description"].str.replace(' wfh ', " work from home ")
    df["description"]=df["description"].str.replace('^wfh ', " work from home ")
    df["description"]=df["description"].str.replace(' wfh$', " work from home ")

    df["description"]=df["description"].str.replace(' bckyrd ', " backyard ")
    df["description"]=df["description"].str.replace('^bckyrd ', " backyard ")
    df["description"]=df["description"].str.replace(' bckyrd$', " backyard ")

    df["description"]=df["description"].str.replace(' busses ', " buses ")
    df["description"]=df["description"].str.replace('^busses ', " buses ")
    df["description"]=df["description"].str.replace(' busses$', " buses ")

    df["description"]=df["description"].str.replace(' two story ', " two storey ")
    df["description"]=df["description"].str.replace('^two story ', " two storey ")
    df["description"]=df["description"].str.replace(' two story$', " two storey ")

    df["description"]=df["description"].str.replace(' nyc ', " new york city ")
    df["description"]=df["description"].str.replace('^nyc ', " new york city ")
    df["description"]=df["description"].str.replace(' nyc$', " new york city ")

    df["description"]=df["description"].str.replace(' xtra ', " extra ")
    df["description"]=df["description"].str.replace('^xtra ', " extra ")
    df["description"]=df["description"].str.replace(' xtra$', " extra ")

    df["description"]=df["description"].str.replace(' opprtnty ', " opportunity ")
    df["description"]=df["description"].str.replace('^opprtnty ', " opportunity ")
    df["description"]=df["description"].str.replace(' opprtnty$', " opportunity ")

    df["description"]=df["description"].str.replace(' nghbrhd ', " neighbourhood ")
    df["description"]=df["description"].str.replace('^nghbrhd ', " neighbourhood ")
    df["description"]=df["description"].str.replace(' nghbrhd$', " neighbourhood ")

    df["description"]=df["description"].str.replace(' cncpt ', " concept ")
    df["description"]=df["description"].str.replace('^cncpt ', " concept ")
    df["description"]=df["description"].str.replace(' cncpt$', " concept ")

    df["description"]=df["description"].str.replace(' expsr ', " exposure ")
    df["description"]=df["description"].str.replace('^expsr ', " exposure ")
    df["description"]=df["description"].str.replace(' expsr$', " exposure ")

    df["description"]=df["description"].str.replace(' wdw ', " window ")
    df["description"]=df["description"].str.replace('^wdw ', " window ")
    df["description"]=df["description"].str.replace(' wdw$', " window ")

    df["description"]=df["description"].str.replace(' entrce ', " entrance ")
    df["description"]=df["description"].str.replace('^entrce ', " entrance ")
    df["description"]=df["description"].str.replace(' entrce$', " entrance ")

    df["description"]=df["description"].str.replace(' xl ', " extra large ")
    df["description"]=df["description"].str.replace('^xl ', " extra large ")
    df["description"]=df["description"].str.replace(' xl$', " extra large ")

    df["description"]=df["description"].str.replace(' equip ', " equipment ")
    df["description"]=df["description"].str.replace('^equip ', " equipment ")
    df["description"]=df["description"].str.replace(' equip$', " equipment ")

    df["description"]=df["description"].str.replace(' blk ', " black ")
    df["description"]=df["description"].str.replace('^blk ', " black ")
    df["description"]=df["description"].str.replace(' blk$', " black ")

    df["description"]=df["description"].str.replace(' sytem ', " system ")
    df["description"]=df["description"].str.replace('^sytem ', " system ")
    df["description"]=df["description"].str.replace(' sytem$', " system ")

    df["description"]=df["description"].str.replace(' hook up ', " hookup ")
    df["description"]=df["description"].str.replace('^hook up ', " hookup ")
    df["description"]=df["description"].str.replace(' hook up$', " hookup ")

    df["description"]=df["description"].str.replace(' elf ', " electric lights fixtures ")
    df["description"]=df["description"].str.replace('^elf ', " electric lights fixtures ")
    df["description"]=df["description"].str.replace(' elf$', " electric lights fixtures ")

    ## Remove the phone number

    df["description"]=df["description"].str.replace(r'\d\d\d\s?\d\d\d\s?\d\d\d\d', " ")

    ## convert " k " to " thousand "
    df["description"]=df["description"].str.replace(' k ', " thousand ")

    ## remove more than 1 spaces
    df["description"]=df["description"].str.replace("\s\s+", " ")


    ## remove single character except "a" or number again
    df["description"]=df["description"].str.replace(r'\s[bcdefghijklmnopqrstuvwxyzBCDEFGHIJKLMNOPQRSTUVWXYZ]\s', " ")

    ## df["description"]=df["description"].str.replace(r'\d+', " ")

    ## the word and abbrivation changed too much
    ## spelling correction using TextBlob will not be used 

    ## tokenization again
    tokeniser = RegexpTokenizer(r'\w+')
    df["description"]=df["description"].apply(lambda x: tokeniser.tokenize(x))

    ## convert words to numbers
    from word2number import w2n
    def numtoword(text_tokens):
        new_list=[]
        for word in text_tokens:
            try: 
                word=w2n.word_to_num(word)
                new_list.append(word)
            except: 
                new_list.append(word)
        return new_list

    df['description'] = df['description'].apply(lambda x: numtoword(x))


    ## Lemmatisation
    lemmatiser = WordNetLemmatizer()
    df["description"]=df["description"].apply(lambda x: [lemmatiser.lemmatize(str(word)) for word in x])

    return df

class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(next(iter(word2vec.values())))
        

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0) for words in X])

def word2vec(df):
    model=Word2Vec.load(output_folder+"/w2v.model")
    w2v = dict(zip(model.wv.index2word, model.wv.syn0))
    MEV=MeanEmbeddingVectorizer(w2v)
    w2v_df=MEV.transform(df)
    return w2v_df

def predict_keras(desc):
    model2 = load_model(output_folder+"/"+'finalized_model_description_only.h5')
    predicted_price=model2.predict(desc)
    scaler = joblib.load(output_folder+"/minmax_scaler.gz")
    price=scaler.inverse_transform(predicted_price)
    output_price=price[0][0]
    return output_price