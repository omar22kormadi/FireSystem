date = '2019-01-01'

# split date into year, month, and day
year, month, day = date.split('-')

print(f"Year: {year}, Month: {month}, Day: {day}")
print(f"Year: {type(year)}, Month: {type(month)}, Day: {type(int(day))}")