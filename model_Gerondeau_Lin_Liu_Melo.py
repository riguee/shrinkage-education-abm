import geopandas as gpd
import json
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
from mesa.datacollection import DataCollector
from mesa import Model
from mesa.time import BaseScheduler, RandomActivation
from mesa_geo.geoagent import GeoAgent, AgentCreator
from mesa_geo import GeoSpace
from shapely.geometry import Point
import shapely
import random
import scipy.stats as stats
from scipy.spatial import cKDTree
from shapely.geometry import Point



def random_points_within(poly, num_points):
	min_x, min_y, max_x, max_y = poly.bounds

	points = []

	while len(points) < num_points:
		random_point = Point([random.uniform(min_x, max_x), random.uniform(min_y, max_y)])
		if (random_point.within(poly)):
			points.append(random_point)

	return points

def ckdnearest(gdA, gdB):

	nA = np.array(list(gdA.geometry.apply(lambda x: (x.x, x.y))))
	nB = np.array(list(gdB.geometry.apply(lambda x: (x.x, x.y))))
	btree = cKDTree(nB)
	dist, idx = btree.query(nA, k=1)
	gdB_nearest = gdB.iloc[idx].drop(columns="geometry").reset_index(drop=True)
	gdf = pd.concat(
		[
			gdA.reset_index(drop=True),
			gdB_nearest,
			pd.Series(dist, name='dist_closest_school')
		],
		axis=1)

	return gdf

def f(x, std, avg):
	return 1-(stats.norm.cdf(x,loc=avg, scale=std*0.9))

class Zip(GeoAgent):
	def __init__(self, unique_id, model, shape):
		super().__init__(unique_id, model, shape)
		self.population = 0
		self.family = 0
		self.non_family = 0
		self.family_size = {2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
		self.non_family_size = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
		self.average_income = 0
		self.average_housing_cost = 0
		self.schools = 0
		self.ethnicities = {}
		self.families_with_children = 0
		self.houses = 0
		self.vacant = 0
		self.households = 0

	def update_values(self, verbose=1):
		if verbose == 2 or verbose == 1:
			print('Start updating zip ', self.unique_id)
		self.reset_count()
		tmp = self.model.grid.agents
		if verbose==2:
			print("There are " ,len(tmp), " agents in the grid.")
		neighbors = []
		tmp_h = 0
		total = 0
		vacant = 0
		for a in tmp:
			if a.shape.within(self.shape):
				if type(a).__name__=="HouseholdAgent":
					neighbors.append(a)
				if type(a).__name__=="School":
					self.schools += 1
				if type(a).__name__=='HouseAgent':
					total+=1
					if a.availability==True or a.status=='Empty':
						vacant+=1
		self.houses=total
		self.vacant = vacant/total
		self.households = len(neighbors)
		if verbose==2 or verbose==1:
			print('There are ', int(len(neighbors)), ' agents in the zip.')
		incomes = []
		housing_costs = []
		races = {}
		children = []
		for n in neighbors:
			if n.children:
				children.append(1)
			else:
				children.append(0)
			if n.race not in races.keys():
				races[n.race]=1
			else:
				races[n.race]+=1
			incomes.append(n.income)
			if n.housing_cost:
				housing_costs.append(n.housing_cost)
			if n.htype == 'family':
				self.family+=1
				self.family_size[n.size]+=1
			else:
				self.non_family+=1
				self.non_family_size[n.size]+=1
		self.families_with_children = np.mean(children)
		self.ethnicities = races
		self.average_income = int(np.mean(incomes))
		self.average_housing_cost = int(np.mean(housing_costs))
		pop_non_fam = sum([i*j for i,j in zip(self.non_family_size.keys(), self.non_family_size.values())])
		pop_fam = sum([i*j for i,j in zip(self.non_family_size.keys(), self.family_size.values())])
		self.population = pop_non_fam+pop_fam
		if verbose == 2:
			print(f"there are {self.population} people")
			print('Values updated.')

	def reset_count(self):
		print('Reseting count for ', self.unique_id, '.')
		self.family = 0
		self.non_family = 0
		self.schools = 0
		self.family_size = {2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
		self.non_family_size = {1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
		self.average_income = 0
		self.average_housing_cost = 0
		self.ethnicities = {}
		self.families_with_children = 0
		self.vacant=0
		self.houses=0
		self.housesolds = 0

class HouseAgent(GeoAgent):
	def __init__(self, unique_id, model, shape):
		super().__init__(unique_id, model, shape)
		self.availability = True
		self.rent = True
		self.status = ''
		self.household = ''
		self.similarity_score = ''
	def move_in(self, household):
		self.availability=False
		self.household=household.hid
#		  household.house=self.id
	def move_out(self):
		self.availability=True
		self.household_id=''



class HouseholdAgent(GeoAgent):
	def __init__(self, unique_id, model, shape):
		super().__init__(unique_id, model, shape)
		self.last_move = 0

	def step(self, sim_thresh=0.3, move_frequency=5, vacancy_threshold=.5):
		#if random.random()>0.4:
		#	self.last_move+=1
		#	return
		#print(self.model.schedule.get_agent_count())
		counter_sim = 0
		counter_school = 0
		counter_move = 0
		counter_vacancy = 0
		counter_income = 0
		schools = []
		all_zip = []
		all_school = []
		available_house = []
		all_households = []
		curr_house=None
		for a in self.model.grid.agents:
			if type(a).__name__=="Zip":
				all_zip.append(a)
				if a.GEOID10==self.zipcode:
					vacant = a.vacant
			if type(a).__name__=='HouseAgent':
				if a.availability==True:
					available_house.append(a)

				if str(a.household)==str(self.hid) or str(a.hid)==str(self.house) or a.shape==self.shape:
					curr_house = a
					#similarity = a.__getattribute__(f"similarity {self.race}")


			if self.children:
				if type(a).__name__=='School':
					all_school.append(a)
					if a.zip==self.zipcode:
						schools.append(a.shape)
			if type(a).__name__=='HouseholdAgent':
				all_households.append(a)
		if curr_house==None:
			#print('Household %s with house %s has no house '%(self.hid, self.house))
			return
		nb_schools=len(schools)
		closest_school=curr_house.dist_closest_school
		#if similarity < sim_thresh:
		#	counter_sim+=1
		if closest_school>self.model.avg_distance_to_school and self.children==True:
			counter_school+=1
		if self.last_move>move_frequency:
			counter_move+=1
		if self.zipcode.vacant>vacancy_threshold:
			counter_vacancy+=1
		# removing the school spending to simulate funding for families
		spending = 10000 + (self.size - 1) * 5000  + self.children * 600 * 12
		spending_no_school = 10000 + (self.size - 1) * 5000
		total_spending = self.housing_cost + spending
		leftover = self.income-total_spending
		money_flag = False
		if leftover<0:
			counter_income+=3
			money_flag==True
		if leftover/self.income<0.05:
			counter_income+=1
			money_flag==True

		all_counter=counter_move+counter_sim+counter_school+counter_income+counter_vacancy
		prob_move=((stats.norm.cdf(all_counter,loc=4, scale=2.43)))
		# if the agent doesn't move, add one to the counter of last move and pass
		if random.random()>prob_move:
			self.last_move+=1
			return
		# otherwise, decision rule for moving

		else:
			potential_houses = available_house.copy()
			potential_zips=	 []
			for h in available_house:
				if h.zipcode.schools==0 and self.children==True:
					potential_houses.remove(h)
				elif h.zipcode.vacant>vacancy_threshold:
					potential_houses.remove(h)
				elif h.zipcode.vacant==0:
					potential_houses.remove(h)
				#elif h.__getattribute__(f"similarity {self.race}")<sim_thresh:
				#	potential_houses.remove(h)
				elif money_flag==True and h.cost>=self.housing_cost:
					potential_houses.remove(h)
				elif self.income - (h.cost + spending)<0:
					potential_houses.remove(h)
				elif self.children==True and h.dist_closest_school>self.model.avg_distance_to_school:
					potential_houses.remove(h)

			if len(potential_houses)<50:
				if self.income>self.model.richest_percentile:
					curr_house.move_out()
					self.model.schedule.remove(self)
					self.zipcode_hid=None
					#print("Agent leaves the city")
					return
				else:
					self.last_move+=1
					return

			if self.children:
				score=0
				for h in potential_houses:
					if h.CompositeIndex>score:
						final_house=h
						score=h.CompositeIndex
			else:
				final_house = random.choice(potential_houses)
			self.zipcode=final_house.zipcode
			self.zipcode_hid=final_house.zipcode
			self.housing_cost=final_house.cost
			self.housing_income_pct=self.housing_cost/self.income
			self.house=final_house.hid
			self.shape=final_house.shape
			curr_house.move_out()
			final_house.move_in(self)


		return None

class School(GeoAgent):
	def __init__(self, unique_id, model, shape):
		super().__init__(unique_id, model, shape)
		tmp = self.model.grid.agents
		self.zip = ''
		for z in tmp:
			if type(z).__name__=="Zip" and self.shape.within(z.shape):
				self.zip = z.unique_id


class GeoModel(Model):
	def compute_ethnic_similarity(self, agents=None, neighbours_table=None, threshold= 200):

		if type(agents).__name__ == 'GeoDataFrame':
			household_df = agents[['hid', 'geometry']].copy()
			household_df = household_df.rename(columns={'hid': 'index'}).set_index('index').set_geometry('geometry')
			if type(neighbours_table).__name__=="GeoDataFrame":
				neighbour_df = neighbours_table[['hid', 'race', 'geometry']].copy()
				neighbour_df = neighbour_df.rename(columns={'hid': 'index', 'race': 'ethnicity_neighbour'}).set_index('index').set_geometry('geometry')
			if type(neighbours_table).__name__!="GeoDataFrame":
				neighbour_df = agents[['hid', 'race', 'geometry']].copy()
				neighbour_df = neighbour_df.rename(columns={'hid': 'index', 'race': 'ethnicity_neighbour'}).set_index('index').set_geometry('geometry')
		else:
			agents = {'index':[] ,'geometry': []}
			neighbours = {'index':[] ,'geometry': [], 'ethnicity_neighbour': []}
			all_agents = self.grid.agents
			if type(neighbours_table).__name__=="GeoDataFrame":
				neighbour_df = neighbours_table[['hid', 'race', 'geometry']].copy()
				neighbour_df = neighbour_df.rename(columns={'hid': 'index', 'race': 'ethnicity_neighbour'}).set_index('index').set_geometry('geometry')
				for a in all_agents:
					if type(a).__name__=='HouseAgent':
						agents['index'].append(a.hid)
						agents['geometry'].append(a.shape)
			else:
				for a in all_agents:
					if type(a).__name__=='HouseAgent':
						agents['index'].append(a.hid)
						agents['geometry'].append(a.shape)
					elif type(a).__name__=='HouseholdAgent':
						neighbours['index'].append(a.hid)
						neighbours['geometry'].append(a.shape)
						neighbours['ethnicity_neighbour'].append(a.race)
				neighbour_df = gpd.GeoDataFrame.from_dict(neighbours).set_index('index').set_geometry('geometry')
				neighbour_df = neighbour_df.set_crs(epsg=self.crs)
			household_df = gpd.GeoDataFrame.from_dict(agents).set_index('index').set_geometry('geometry')
			household_df = household_df.set_crs(epsg=self.crs)
			# a copy of the households -> to consider the neighbours, unless the neighbours are different
		ethnicities = ['White', 'Black', 'American Indian', 'Asian', 'Native Hawaiian', 'Other', 'Two or more races']
		# area around each household where we will consider the neighbours
		disk_df = gpd.GeoDataFrame.from_dict({'index': household_df.index, 'geometry' :[t.buffer(threshold) for t in household_df['geometry']]}).set_index('index').set_geometry('geometry')
		disk_df = disk_df.set_crs(epsg=self.crs)
		# join the neighbours on the households, based on their presence in the radius
		join_df = gpd.sjoin(disk_df, neighbour_df, how="left", op='contains').merge(neighbour_df['geometry'], left_on='index_right', right_index=True)
		join_df = join_df.reset_index()
		# remove the current household from the neighbours
		join_df = join_df.loc[join_df['index']!=join_df['index_right']]
		# remove the radius and add the original points back into the dataframe
		join_df = join_df.drop(columns=['geometry_x']).rename(columns={'geometry_y': 'coord_neighbours'})
		join_df = join_df.merge(household_df[['geometry']], right_index=True, left_on='index')
		# add the distance between points and their neighbours
		join_df['distance'] = join_df.apply(lambda x: x['coord_neighbours'].distance(x['geometry']), axis=1)
		join_df['computed_distance'] = join_df['distance'].apply(lambda x: f(x,50,25))
		denominator_df = join_df[['computed_distance', 'index']].groupby('index').sum()
		final_df = pd.DataFrame(index=join_df[['distance', 'index']].groupby('index').sum().index)
		for e in ethnicities:
			join_df['index_function'] = np.where(join_df['ethnicity_neighbour']==e, 1, 0)
			join_df[f'similarity {e}'] = join_df.apply(lambda x: x['index_function']*x['computed_distance'], axis=1)
			values = join_df[[f'similarity {e}', 'index']].groupby('index').sum().merge(denominator_df, left_index=True, right_index=True)
			final_df[f'similarity {e}'] = values.apply(lambda x: x[f'similarity {e}']/x['computed_distance'], axis=1)

		return final_df


	def __init__(self, type_size, shp, schools, family_races_cumsum, nonfamily_races_cumsum, income_race_cumsum, housing_tenure_cumsum, vacancies, crs=4269):

		self.schedule = RandomActivation(self)
		self.grid = GeoSpace()
#		  shp.crs= "epsg:"+str(crs)
		print("Creating zips.")
		zip_agent_kwargs = dict(model=self)
#		  AC = AgentCreator(agent_class=Zip, agent_kwargs=zip_agent_kwargs)
		AC = AgentCreator(Zip, {"model": self})
		zip_agents = AC.from_GeoDataFrame(gdf=shp)
		self.grid.add_agents(zip_agents)
		print("Done creating zips.")
		print("Creating schools.")
		school_agent_kwargs = dict(model=self)
		school_AC = AgentCreator(School, {"model": self})
		school_agents = school_AC.from_GeoDataFrame(gdf=schools)
		self.grid.add_agents(school_agents)
		self.crs=crs
		print("Done creating schools.")
		ac_population = AgentCreator(HouseholdAgent, {"model": self})
		ac_houses = AgentCreator(HouseAgent, {"model": self})
		agents_df = {
			'hid':[],
			'htype':[],
			'size':[],
			'children': [],
			'tenure':[],
			'race':[],
			'income':[],
			'zipcode': [],
			'zipcode_hid': [],
			'house': [],
			'housing_income_pct': [],
			'housing_cost': [],
			'geometry':gpd.GeoSeries([j for i in [random_points_within(z.shape, type_size["HOUSEHOLD TYPE BY HOUSEHOLD SIZE: Total:"].loc[int(z.unique_id)]) for z in zip_agents] for j in i])
					}
		houses_df = {
			'hid':[],
			'zipcode': [],
			'zipcode_hid': [],
			'status': [],
			'availability' : [],
			'rent': [],
			'sale': [],
			'cost': [],
			'household': [],
			'geometry': agents_df['geometry']
		}
		agents_df['geometry'].crs="epsg:"+str(crs)
		houses_df['geometry'].crs="epsg:"+str(crs)

		#function to match the income to the income brackets of the monthly spending table
		def return_income_range(income):
			if income < 20000:
				return 'less_than_20000'
			elif income >= 20000 and income < 35000:
				return '20000_to_34999'
			elif income >= 35000 and income < 50000:
				return '35000_to_49999'
			elif income >= 50000 and income < 75000:
				return '50000_to_74999'
			else:
				return '75000_and_more'

		# we create the households associated to each zip code
		for z in zip_agents:
			print(f"Creating agents linked to zip {z.unique_id}")
			type_size_data = type_size.loc[int(z.unique_id)]
			for i in range(type_size_data["HOUSEHOLD TYPE BY HOUSEHOLD SIZE: Total:"]):
				coords = z.shape.representative_point()
				agents_df['zipcode'].append(z)
				agents_df['zipcode_hid'].append(z.unique_id)
				if i < type_size_data['HOUSEHOLD TYPE BY HOUSEHOLD SIZE: Family households:']:
					fam_data = type_size_data.iloc[2:8]
					cum_sum = np.insert(fam_data.values.cumsum(), 0, 0)
					agents_df['htype'].append('family')
					counter = 0
					while i >= cum_sum[counter]:
						counter += 1
					agents_df['size'].append(counter + 1)
					agents_df['children'].append(random.random()<1-(type_size_data['HOUSEHOLDS AND FAMILIES: Total!!Estimate!!AGE OF OWN CHILDREN!!Households with own children under 18 years']/type_size_data['HOUSEHOLDS AND FAMILIES: Total!!Estimate!!FAMILIES!!Total families'])**(counter + 1))
					if random.random()<type_size_data['HOUSEHOLDS AND FAMILIES: Total!!Estimate!!HOUSING TENURE!!Owner-occupied housing units']/100:
						curr_tenure = 'owner'
						agents_df['tenure'].append('owner')
					else:
						curr_tenure = 'renter'
						agents_df['tenure'].append('renter')
					val = random.random()
					counter = 0
					if val <family_races_cumsum.loc[int(z.unique_id)].values[0]:
						curr_race = family_races_cumsum.columns[0]
						agents_df['race'].append(curr_race)
					else:
						while val >= family_races_cumsum.loc[int(z.unique_id)].values[counter]:
							counter+=1
						curr_race = family_races_cumsum.columns[counter]
						agents_df['race'].append(curr_race)
				else:
					fam_data = type_size_data.iloc[9:16]
					cum_sum = np.insert(fam_data.values.cumsum(), 0, 0)
					agents_df['htype'].append('non-family')
					j = i-type_size_data['HOUSEHOLD TYPE BY HOUSEHOLD SIZE: Family households:']
					counter = 0
					while j >= cum_sum[counter]:
							counter += 1
					agents_df['size'].append(counter)
					agents_df['children'].append(random.random()<1-(type_size_data['HOUSEHOLDS AND FAMILIES: Nonfamily household!!Estimate!!AGE OF OWN CHILDREN!!Households with own children under 18 years']/type_size_data['HOUSEHOLDS AND FAMILIES: Nonfamily household!!Estimate!!Total households'])**counter)
					if random.random()<type_size_data['HOUSEHOLDS AND FAMILIES: Nonfamily household!!Estimate!!HOUSING TENURE!!Owner-occupied housing units']/100:
						curr_tenure = 'owner'
						agents_df['tenure'].append('owner')
					else:
						curr_tenure = 'renter'
						agents_df['tenure'].append('renter')
					val = random.random()
					counter = 0
					if val <nonfamily_races_cumsum.loc[int(z.unique_id)].values[0]:
						curr_race = nonfamily_races_cumsum.columns[0]
						agents_df['race'].append(curr_race)
					else:
						while val >= nonfamily_races_cumsum.loc[int(z.unique_id)].values[counter]:
							counter+=1
						curr_race = nonfamily_races_cumsum.columns[counter]
						agents_df['race'].append(curr_race)
				incomes = income_race_cumsum['White'].columns
				income_rand = random.random()
				income_race_cumsum_curr = income_race_cumsum[curr_race].loc[int(z.unique_id)].values
				if income_rand < income_race_cumsum_curr[0]:
					curr_income = incomes[0]
					agents_df['income'].append(curr_income)
				else :
					counter_income = 0
					while income_rand>=income_race_cumsum_curr[counter_income]:
						counter_income+=1
					curr_income = incomes[counter_income]
					agents_df['income'].append(curr_income)
				income_range = return_income_range(int(curr_income))
				housing_rand = random.random()
				housing_tenure_income_curr = housing_tenure_cumsum[curr_tenure][income_range].loc[int(z.unique_id)].values
				if housing_rand < housing_tenure_income_curr[0]:
					agents_df['housing_income_pct'].append(0.19)
					curr_housing_cost=0.19*int(curr_income)
					agents_df['housing_cost'].append(0.19*int(curr_income))
				elif housing_rand < housing_tenure_income_curr[1]:
					agents_df['housing_income_pct'].append(0.29)
					curr_housing_cost=0.29*int(curr_income)
					agents_df['housing_cost'].append(0.29*int(curr_income))
				else:
					agents_df['housing_income_pct'].append(0.40)
					curr_housing_cost=0.4*int(curr_income)
					agents_df['housing_cost'].append(0.40*int(curr_income))
				hid = int(str(z.unique_id)+str(i))
				agents_df['hid'].append(hid)
				agents_df['house'].append(hid)


				#also define the inhabited houses and flats
				houses_df['hid'].append(hid)
				houses_df['availability'].append(False)
				houses_df['household'].append(hid)
				houses_df['zipcode_hid'].append(int(z.unique_id))
				houses_df['zipcode'].append(z)
				houses_df['cost'].append(curr_housing_cost)
				if curr_tenure == 'renter':
					houses_df['status'].append('Rent')
					houses_df['rent'].append(True)
					houses_df['sale'].append(False)
				else:
					houses_df['status'].append('Sale')
					houses_df['rent'].append(False)
					houses_df['sale'].append(True)

		household_gdf = gpd.GeoDataFrame(agents_df)

		household_agents = ac_population.from_GeoDataFrame(gdf=household_gdf)
		self.grid.add_agents(household_agents)
		for a in household_agents:
			self.schedule.add(a)
		print("Done creating all households.")



		print("\nCreating empty houses")
		for z in zip_agents:
			print(f"Creating empty houses linked to Zip {z.unique_id}")
			curr_vacancies = vacancies.loc[int(z.unique_id)].values
			print(f"There are {sum(curr_vacancies)} houses to create.")
			geometries = gpd.GeoSeries(random_points_within(z.shape, sum(curr_vacancies)))
			houses_df['geometry'] = pd.concat([houses_df['geometry'], geometries]).reset_index(drop=True)
			disks = gpd.GeoDataFrame([i.buffer(1000) for i in geometries.values]).rename(columns={0:'geometry'}).set_geometry('geometry')
			disks.crs= "epsg:"+str(crs)
			costs = gpd.sjoin(disks, household_gdf, how="left", op='contains').reset_index().groupby('index')['housing_cost'].mean()
			houses_df['cost'] += list(costs)
			for i in range(sum(curr_vacancies)):
				geometry = geometries[i]
				houses_df['household'].append('')
				houses_df['zipcode_hid'].append(int(z.unique_id))
				houses_df['zipcode'].append(z)
				houses_df['hid'].append(str(z.unique_id)+str(i)+'e')
				if i < curr_vacancies[0]:
					houses_df['availability'].append(True)
					houses_df['status'].append('Rent')
					houses_df['rent'].append(True)
					houses_df['sale'].append(False)
				elif i < curr_vacancies[1]:
					houses_df['availability'].append(True)
					houses_df['status'].append('All')
					houses_df['rent'].append(True)
					houses_df['sale'].append(True)
				elif i < curr_vacancies[2]:
					houses_df['availability'].append(True)
					houses_df['status'].append('Sale')
					houses_df['rent'].append(False)
					houses_df['sale'].append(True)
				else:
					houses_df['availability'].append(False)
					houses_df['status'].append('Empty')
					houses_df['rent'].append(False)
					houses_df['sale'].append(False)
#		  for k in houses_df.keys():
#			  print(len(houses_df[k]))
		house_gdf = gpd.GeoDataFrame(houses_df, geometry=houses_df['geometry'])
		#similarity = self.compute_ethnic_similarity(agents=house_gdf, neighbours_table=household_gdf)
		closest_schools = ckdnearest(house_gdf[['hid', 'geometry']], schools[['AttendanceRate', 'CompositeIndex', 'geometry']].reset_index(drop=False).rename(columns={"index": 'school_id'})).set_index('hid')
		self.avg_distance_to_school = np.mean(closest_schools['dist_closest_school'].values)
		self.richest_percentile = np.percentile(household_gdf['income'], 60)
		#house_gdf = house_gdf.merge(similarity, left_on='hid', right_index=True)
		house_gdf = house_gdf.merge(closest_schools[['AttendanceRate', 'CompositeIndex', 'dist_closest_school']], left_on='hid', right_index=True)
		house_gdf.crs= "epsg:"+str(crs)
		houses = ac_houses.from_GeoDataFrame(gdf=house_gdf)
		self.grid.add_agents(houses)
		print("Done creating empty houses.")

		print("\nStart updating zip values.")
		for z in zip_agents:
			z.update_values()
		print("\nThe model is initialized.")

		self.datacollector = DataCollector(
				{
					"summary": get_population,
				}
			)
		self.steps=0
		self.datacollector.collect(self)

	def step(self):
		self.steps += 1
		print(f'step {self.steps}')
		self.schedule.step()
		self.grid._recreate_rtree()

		self.datacollector.collect(self)



def get_population(model):
	agents = model.schedule.agents
	populations = {'total': model.schedule.get_agent_count()}
	ethnicities = {}
	incomes = {}
	families = {}
	children = {}
	for a in agents:
		if type(a).__name__=='HouseholdAgent':
			if type(a.zipcode_hid).__name__=='Zip':
				a.zipcode_hid=a.zipcode_hid.unique_id
			if a.htype=='family':
				if a.zipcode_hid in families.keys():
					families[a.zipcode_hid]+=1
			if a.children:
				if a.children in children.keys():
					children[a.zipcode_hid]+=1

			if a.zipcode_hid in populations.keys():
				populations[a.zipcode_hid]+=a.size
				incomes[a.zipcode_hid].append(a.income)
				if a.race in ethnicities[a.zipcode_hid].keys():
					ethnicities[a.zipcode_hid][a.race]+=a.size
				else:
					ethnicities[a.zipcode_hid][a.race]=a.size
			else:
				if a.htype=='family':
					families[a.zipcode_hid]=1
				if a.children:
					children[a.zipcode_hid]=1
				populations[a.zipcode_hid]=a.size
				incomes[a.zipcode_hid] = [a.income]
				ethnicities[a.zipcode_hid]={}
				ethnicities[a.zipcode_hid][a.race]=a.size
	return populations, incomes, ethnicities, families, children

income_race = pd.read_csv('input/income_per_race.csv').set_index('Unnamed: 0')
income_race = income_race.drop(columns=['state_x', 'state_y'])
shp = gpd.read_file('input/zip_detroit.shp').set_index('index')
shp = shp.drop(columns=['computed_a', 'computed_1','computed_2'])
households = pd.read_csv('input/household_size_type.csv').set_index("zip code tabulation area").drop(columns=["state"])
households = ((households/100)+1).astype(int)
household_families_df = pd.read_csv('input/household_families_subject_table.csv').set_index("zip code tabulation area")
tmp_df = households.merge(household_families_df, left_index=True, right_index=True)
races_df = pd.read_csv('input/household_by_race.csv').set_index('index')
family_households_races_df = races_df[['HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (WHITE ALONE): Family households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (BLACK OR AFRICAN AMERICAN ALONE): Family households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (AMERICAN INDIAN AND ALASKA NATIVE ALONE): Family households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (ASIAN ALONE): Family households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (NATIVE HAWAIIAN AND OTHER PACIFIC ISLANDER ALONE): Family households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (SOME OTHER RACE ALONE): Family households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (TWO OR MORE RACES): Family households:']]
nonfamily_households_races_df = races_df[['HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (WHITE ALONE): Nonfamily households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (BLACK OR AFRICAN AMERICAN ALONE): Nonfamily households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (AMERICAN INDIAN AND ALASKA NATIVE ALONE): Nonfamily households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (ASIAN ALONE): Nonfamily households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (NATIVE HAWAIIAN AND OTHER PACIFIC ISLANDER ALONE): Nonfamily households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (SOME OTHER RACE ALONE): Nonfamily households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (TWO OR MORE RACES): Nonfamily households:']]
family_households_races_df['total'] = family_households_races_df.sum(axis=1)
nonfamily_households_races_df['total'] = nonfamily_households_races_df.sum(axis=1)
family_race_pct = pd.DataFrame(index=family_households_races_df.index)
nonfamily_race_pct = pd.DataFrame(index=nonfamily_households_races_df.index)
columns_family = ['HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (WHITE ALONE): Family households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (BLACK OR AFRICAN AMERICAN ALONE): Family households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (AMERICAN INDIAN AND ALASKA NATIVE ALONE): Family households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (ASIAN ALONE): Family households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (NATIVE HAWAIIAN AND OTHER PACIFIC ISLANDER ALONE): Family households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (SOME OTHER RACE ALONE): Family households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (TWO OR MORE RACES): Family households:']
columns_nonfamily = ['HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (WHITE ALONE): Nonfamily households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (BLACK OR AFRICAN AMERICAN ALONE): Nonfamily households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (AMERICAN INDIAN AND ALASKA NATIVE ALONE): Nonfamily households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (ASIAN ALONE): Nonfamily households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (NATIVE HAWAIIAN AND OTHER PACIFIC ISLANDER ALONE): Nonfamily households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (SOME OTHER RACE ALONE): Nonfamily households:',
  'HOUSEHOLD TYPE (INCLUDING LIVING ALONE) (TWO OR MORE RACES): Nonfamily households:']
races = ['White', 'Black', 'American Indian', 'Asian', 'Native Hawaiian', 'Other', 'Two or more races']
for fam, nonfam,j in zip(columns_family, columns_nonfamily, races):
	family_race_pct[j] = family_households_races_df[fam]/family_households_races_df['total']
	nonfamily_race_pct[j] = nonfamily_households_races_df[nonfam]/nonfamily_households_races_df['total']

family_races_cumsum = pd.DataFrame(family_race_pct.values.cumsum(axis=1), columns=races, index=family_race_pct.index)
nonfamily_races_cumsum = pd.DataFrame(nonfamily_race_pct.values.cumsum(axis=1), columns=races, index=nonfamily_race_pct.index)
family_races_cumsum = pd.DataFrame(family_race_pct.values.cumsum(axis=1), columns=races, index=family_race_pct.index)
nonfamily_races_cumsum = pd.DataFrame(nonfamily_race_pct.values.cumsum(axis=1), columns=races, index=nonfamily_race_pct.index)
incomes = [5000, 12500, 17500, 22500, 27500, 32500, 37500, 42500, 47500, 55000, 67500, 87500, 112500, 137500, 175000, 250000]
income_race_pct = {}
for i,j in enumerate(races):
	income_race_pct[j] = pd.DataFrame(index=income_race.index)
	for k,l in zip(income_race.columns[(i*17)+1:(i+1)*17], incomes):
		total = income_race.iloc[:,(i*17)+1:(i+1)*17].sum(axis=1)
		income_race_pct[j][l] = income_race[k]/total
income_race_cumsum = {}
for i in races:
	income_race_cumsum[i]=income_race_pct[i].cumsum(axis=1)
housing_tenure_income = pd.read_csv("input/housing_tenure_income.csv").set_index('zip code tabulation area').drop(columns=['state'])
def preprocess_housing(housing):
	outcome = {}
	totals=pd.DataFrame(index=housing.index)
	for i,j in enumerate(['less_than_20000', '20000_to_34999', '35000_to_49999', '50000_to_74999', '75000_and_more']):
		totals[f"owner_{j}"] = housing[housing.columns[2+i*4]]
		totals[f"renter_{j}"] = housing[housing.columns[24+i*4]]
	outcome['owner']={}
	outcome['owner']['less_than_20000'] = housing[[
			'Tenure by Housing Costs As A Percentage of Household Income: Owner-occupied housing units:!!Less than $20,000:!!Less than 20 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Owner-occupied housing units:!!Less than $20,000:!!20 to 29 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Owner-occupied housing units:!!Less than $20,000:!!30 percent or more'
	]].divide(totals["owner_less_than_20000"], axis=0)
	outcome['owner']['20000_to_34999'] = housing[[
			'Tenure by Housing Costs As A Percentage of Household Income: Owner-occupied housing units:!!$20,000 to $34,999:!!Less than 20 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Owner-occupied housing units:!!$20,000 to $34,999:!!20 to 29 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Owner-occupied housing units:!!$20,000 to $34,999:!!30 percent or more'
	]].divide(totals["owner_20000_to_34999"], axis=0)
	outcome['owner']['35000_to_49999'] = housing[[
			'Tenure by Housing Costs As A Percentage of Household Income: Owner-occupied housing units:!!$35,000 to $49,999:!!Less than 20 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Owner-occupied housing units:!!$35,000 to $49,999:!!20 to 29 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Owner-occupied housing units:!!$35,000 to $49,999:!!30 percent or more'
	]].divide(totals["owner_35000_to_49999"], axis=0)
	outcome['owner']['50000_to_74999'] = housing[[
			'Tenure by Housing Costs As A Percentage of Household Income: Owner-occupied housing units:!!$50,000 to $74,999:!!Less than 20 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Owner-occupied housing units:!!$50,000 to $74,999:!!20 to 29 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Owner-occupied housing units:!!$50,000 to $74,999:!!30 percent or more'
	]].divide(totals["owner_50000_to_74999"], axis=0)
	outcome['owner']['75000_and_more'] = housing[[
			'Tenure by Housing Costs As A Percentage of Household Income: Owner-occupied housing units:!!$75,000 or more:!!Less than 20 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Owner-occupied housing units:!!$75,000 or more:!!20 to 29 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Owner-occupied housing units:!!$75,000 or more:!!30 percent or more'
	]].divide(totals["owner_75000_and_more"], axis=0)

	outcome['renter']={}
	outcome['renter']['less_than_20000'] = housing[[
			'Tenure by Housing Costs As A Percentage of Household Income: Renter-occupied housing units:!!Less than $20,000:!!Less than 20 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Renter-occupied housing units:!!Less than $20,000:!!20 to 29 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Renter-occupied housing units:!!Less than $20,000:!!30 percent or more'
	]].divide(totals["renter_less_than_20000"], axis=0)
	outcome['renter']['20000_to_34999'] = housing[[
			'Tenure by Housing Costs As A Percentage of Household Income: Renter-occupied housing units:!!$20,000 to $34,999:!!Less than 20 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Renter-occupied housing units:!!$20,000 to $34,999:!!20 to 29 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Renter-occupied housing units:!!$20,000 to $34,999:!!30 percent or more'
	]].divide(totals["renter_20000_to_34999"], axis=0)
	outcome['renter']['35000_to_49999'] = housing[[
			'Tenure by Housing Costs As A Percentage of Household Income: Renter-occupied housing units:!!$35,000 to $49,999:!!Less than 20 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Renter-occupied housing units:!!$35,000 to $49,999:!!20 to 29 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Renter-occupied housing units:!!$35,000 to $49,999:!!30 percent or more'
	]].divide(totals["renter_35000_to_49999"], axis=0)
	outcome['renter']['50000_to_74999'] = housing[[
			'Tenure by Housing Costs As A Percentage of Household Income: Renter-occupied housing units:!!$50,000 to $74,999:!!Less than 20 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Renter-occupied housing units:!!$50,000 to $74,999:!!20 to 29 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Renter-occupied housing units:!!$50,000 to $74,999:!!30 percent or more'
	]].divide(totals["renter_50000_to_74999"], axis=0)
	outcome['renter']['75000_and_more'] = housing[[
			'Tenure by Housing Costs As A Percentage of Household Income: Renter-occupied housing units:!!$75,000 or more:!!Less than 20 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Renter-occupied housing units:!!$75,000 or more:!!20 to 29 percent',
		   'Tenure by Housing Costs As A Percentage of Household Income: Renter-occupied housing units:!!$75,000 or more:!!30 percent or more'
	]].divide(totals["renter_75000_and_more"], axis=0)

	return outcome
housing_tenure_dict = preprocess_housing(housing_tenure_income)
housing_tenure_cumsum = {}
for i in housing_tenure_dict.keys():
	housing_tenure_cumsum[i] = {}
	for j in housing_tenure_dict[i].keys():
		housing_tenure_cumsum[i][j] = housing_tenure_dict[i][j].cumsum(axis=1)
vacancies = pd.read_csv('input/vacancies.csv').set_index('zip code tabulation area').drop(columns=['state'])
vacancies_formatted = pd.DataFrame(index=vacancies.index)
vacancies_formatted['Rent'] = vacancies['Vacancy Status: For rent']
vacancies_formatted['Other'] = vacancies['Vacancy Status: Other vacant']
vacancies_formatted['Sale'] = vacancies['Vacancy Status: For sale only']
vacancies_formatted['Empty'] = vacancies[['Vacancy Status: Rented, not occupied', 'Vacancy Status: Sold, not occupied', 'Vacancy Status: For seasonal, recreational, or occasional use']].sum(axis=1)
vacancies_formatted = ((vacancies_formatted/100)+1).astype(int)

schools_final = gpd.read_file("input/new_generated_schools.shp").rename(columns={'CompositeI':'CompositeIndex', 'Attendance': 'AttendanceRate'})
#for the initial data without change, remove the last 20 rows
schools_final = schools_final.iloc[:-20, :]


#improve school scores
#def improve_score(x):
#	if x<30:
#		return x+20
#	else:
#		return x
#schools_final['CompositeIndex'] = schools_final['CompositeIndex'].apply(improve_score)


tmp_m = GeoModel(tmp_df, shp.iloc[:,:], schools_final, family_races_cumsum, nonfamily_races_cumsum, income_race_cumsum, housing_tenure_cumsum, vacancies_formatted, crs=3857)

print(f"\n\n\nThe model initially has {tmp_m.schedule.get_agent_count()} households.")

def print_map(tmp_m, show=1, save=1):
	dicty = {'type':[], 'geometry':[]}
	schools_dicty = {'index': [], 'geometry':[]}
	zipsies = {'index': [], 'geometry': []}
	for i in tmp_m.grid.agents:
	    if type(i).__name__=='HouseAgent':
	        if i.availability:
	            dicty['type'].append('available house')
	        else:
	            dicty['type'].append('occupied house')
	        dicty['geometry'].append(i.shape)
	    if type(i).__name__=='School':
	        schools_dicty['index'].append(i.unique_id)
	        schools_dicty['geometry'].append(i.shape)
	    if type(i).__name__=='Zip':
	        zipsies['index'].append(i.unique_id)
	        zipsies['geometry'].append(i.shape)
	dicty_gdf = gpd.GeoDataFrame.from_dict(dicty, geometry='geometry')
	schools_dicty_gdf = gpd.GeoDataFrame.from_dict(schools_dicty, geometry='geometry')
	zipsies_gdf = gpd.GeoDataFrame.from_dict(zipsies, geometry='geometry')

	fig, ax = plt.subplots(figsize=(20,20))
	dicty_gdf.plot(column='type', ax=ax, legend=True, alpha=.2, categorical=True, cmap=matplotlib.cm.get_cmap('viridis').reversed())
	schools_dicty_gdf.plot(ax=ax, color = 'red', label='schools')
	zipsies_gdf.boundary.plot(ax=ax, label='zip')
	fig.legend()
	plt.title(f'State of the model at step {tmp_m.steps}')
	ax.axis('off')
	plt.tight_layout()
	if save==1:
		plt.savefig(f'output/model_step_{tmp_m.steps}.png')
	if show==1:
		plt.show()

print_map(tmp_m, save=1)

print("Start model steps.")
for i in range(10):
	tmp_m.step()
	print(f"The model contains {tmp_m.schedule.get_agent_count()} households at this step.")

	#similarity = tmp_m.compute_ethnic_similarity()
	#cols = similarity.columns
	for a in tmp_m.grid.agents:
		# if type(a).__name__=='HouseAgent':
			# for c in cols:
				# try:
					# a.__setattr__(c, similarity[c].loc[a.hid])
				# except:
					# a.__setattr__(c, np.mean(similarity[c].values))
		if type(a).__name__=="Zip":
			a.update_values(verbose=0)

print_map(tmp_m, show=1, save=1)
summary_df = tmp_m.datacollector.get_model_vars_dataframe()
#summary_df.to_csv('summary_df_init.csv')
summary_df = pd.DataFrame(summary_df['summary'].tolist(), index=summary_df.index)
for i in summary_df.columns:
	summary_df = summary_df.merge(summary_df[i].dropna().apply(pd.Series), left_index=True, right_index=True)
summary_df.to_csv('output/summary_dataframe.csv')
