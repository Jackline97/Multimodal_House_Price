def Data_preprocessing(df):
    scaler = MinMaxScaler()

    # Numerical Feature
    numerical_features = ["bedroom", "bedroomAboveGrade", "bedroomBelowGrade", "bathroom", "bathroomTotal",
                          "bathroomPartial",
                          "totalParkingSpaces", "storeys", "maintenanceFees", 'landSize']

    X_num = df[numerical_features]
    X_num = scaler.fit_transform(X_num)
    X_num = pd.DataFrame(X_num)
    df[numerical_features] = X_num
    df["longitude"] = df["longitude"] * 0.01
    df["latitude"] = df["latitude"] * 0.01
    numerical_features += ['latitude', 'longitude']
    # Boolean Feature
    boolean_features = ['parkingAttachedGarage',
                        'parkingUnderground', 'parkingInsideEntry', 'parkingSurfaced',
                        'parkingOversize', 'parkingGravel', 'parkingGarage', 'parkingShared',
                        'parkingDetachedGarage', 'parkingCarport', 'parkingInterlocked',
                        'parkingVisitorParking', 'amenityClubhouse', 'amenityCarWash', 'amenityMusicRoom',
                        'amenityStorageLocker', 'amenitySauna', 'amenityPartyRoom',
                        'amenityRecreationCentre', 'amenityGuestSuite', 'amenityFurnished',
                        'amenityLaundryFacility', 'amenityExerciseCentre',
                        'amenityLaundryInSuite', 'amenitySecurity', 'amenityWhirlpool',
                        'efinishWood', 'efinishBrick', 'efinishHardboard', 'efinishWoodsiding',
                        'efinishLog', 'efinishMetal', 'efinishSteel', 'efinishStone',
                        'efinishWoodshingles', 'efinishStucco', 'efinishSiding',
                        'efinishConcrete', 'efinishShingles', 'efinishAluminumsiding',
                        'efinishCedarshingles', 'efinishVinyl', 'efinishVinylsiding',
                        'featurePetNotAllowed', 'AirportNearby',
                        'GolfNearby', 'MarinaNearby', 'ShoppingNearby', 'WaterNearby',
                        'WorshipPlaceNearby', 'RecreationNearby', 'PlaygroundNearby',
                        'PublicTransitNearby', 'ParkNearby', 'SchoolsNearby', 'HospitalNearby',
                        'HighwayNearby', 'SkiAreaNearby']

    # Category Feature
    cate_features = ['city', 'typeBuilding', 'title', 'styleAttach',
                     'cooling', 'basementType', 'basementFinish',
                     'heatingType1', 'heatingType2', 'heatingEnergy1', 'heatingEnergy2',
                     'featureLotSlope', 'featureDriveway', 'featureLotPositionType',
                     'featureOutdoorAreaType', 'featureOutdoorLandscape',
                     'featureAdditionalFacility']

    X_category = df[cate_features]
    for col in cate_features:
        X_category[col] = X_category[col].astype('category')
        X_category[col] = X_category[col].cat.codes
    df[cate_features] = X_category

    # Label Price
    price_range = []

    for price in df["price"]:
        if price < 5e5:
            price_range.append(0)
        elif 5e5 <= price < 15e5:
            price_range.append(1)
        elif 15e5 <= price < 25e5:
            price_range.append(2)
        elif 25e5 <= price < 35e5:
            price_range.append(3)
        elif 35e5 <= price < 80e5:
            price_range.append(4)
        else:
            price_range.append(5)

    #     df = df.reset_index(drop=True)
    df['price_range'] = price_range
    df = df.dropna()
    return df, boolean_features, cate_features, numerical_features

