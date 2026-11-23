"""Free bottle on a bimanual worktable; only robot joints are actuated."""
import math
from copy import deepcopy
from pathlib import Path
import xml.etree.ElementTree as ET
from .mechanism import scene_xml

PANDA = Path(__file__).parent / 'assets' / 'panda'


def robot_scene():
    root = ET.fromstring('''<mujoco model="dual_panda_balance">
      <compiler angle="radian" autolimits="true"/>
      <option timestep="0.002" integrator="implicitfast" cone="elliptic" iterations="100"/>
      <size nconmax="500" njmax="3000"/>
      <visual><global offwidth="960" offheight="720"/><quality shadowsize="2048"/></visual>
      <default><geom friction="1.1 .01 .001" solref=".008 1" solimp=".95 .99 .001"/><joint armature=".02"/></default>
      <asset><texture name="floor_tex" type="2d" builtin="checker" rgb1=".12 .15 .19" rgb2=".16 .19 .23" width="256" height="256"/><material name="floor_mat" texture="floor_tex" texrepeat="5 5"/></asset>
      <worldbody>
        <light pos="0 -1 3" dir="0 0 -1" diffuse=".8 .8 .8"/>
        <light pos="1 1 2" dir="-1 -1 -1" diffuse=".5 .5 .5"/>
        <geom name="floor" type="plane" size="3 3 .1" material="floor_mat"/>
        <geom name="table" type="box" pos="0 0 .76" size=".72 .82 .04" rgba=".28 .32 .38 1"/>
      </worldbody><actuator/><sensor/><contact/>
    </mujoco>''')
    for side, y, color in [('left', -.42, '.1 .55 .95 1'), ('right', .42, '.95 .5 .1 1')]:
        robot = ET.parse(PANDA / 'no_texture_robot.xml').getroot()
        for node in robot.iter():
            for attr in ('name', 'joint', 'mesh', 'material', 'site', 'body'):
                if attr in node.attrib:
                    node.set(attr, side + '_' + node.get(attr))
            if node.tag == 'mesh': node.set('file', str(PANDA / node.get('file')))
        for geom in robot.iter('geom'):
            if geom.get('name', '').endswith('_collision'): geom.set('rgba', '0 0 0 0')
        base = robot.find('worldbody/body')
        base.set('pos', f'-.40 {y} .8')
        hand = base.find(f'.//body[@name="{side}_right_hand"]')
        ET.SubElement(hand, 'site', name=side + '_wrist', size='.008', rgba=color)
        ET.SubElement(hand, 'geom', name=side + '_palm', type='box', pos='0 0 .045', size='.055 .025 .02', rgba=color)
        ET.SubElement(hand, 'site', name=side + '_tip', pos='0 0 .12', size='.004', rgba=color)
        for finger, sign in [('a', -1), ('b', 1)]:
            name = f'{side}_finger_{finger}'
            body = ET.SubElement(hand, 'body', name=name)
            ET.SubElement(body, 'joint', name=name, type='slide', axis=f'{sign} 0 0', range='0 .045', damping='4', armature='.001')
            ET.SubElement(body, 'geom', name=name + '_pad', type='box', pos=f'{sign*.008} 0 .11', size='.008 .016 .035', mass='.04', rgba='.18 .2 .23 1', friction='1.5 .01 .001')
            ET.SubElement(body, 'site', name=name + '_touch', type='box', pos=f'{sign*.001} 0 .11', size='.003 .017 .036', rgba='0 0 0 0')
            ET.SubElement(root.find('sensor'), 'touch', name=name + '_normal', site=name + '_touch')
        for typ in ('force', 'torque'):
            ET.SubElement(root.find('sensor'), typ, name=side + '_' + typ, site=side + '_wrist')
        root.find('worldbody').append(base)
        root.find('asset').extend(robot.find('asset'))
        root.find('actuator').extend(robot.find('actuator'))
    # Finger actuators follow all fourteen arm motors.
    for side in ('left', 'right'):
        for finger in ('a', 'b'):
            name = f'{side}_finger_{finger}'
            ET.SubElement(root.find('actuator'), 'position', name=name, joint=name, kp='400', ctrlrange='0 .045', forcerange='-12 12')
    return root


def hand_wrist_cameras(world):
    # Rigid on each hand, 70 mm along tool -Y, aimed just past the fingertips.
    for side in ('left', 'right'):
        ET.SubElement(world.find(f".//body[@name='{side}_right_hand']"), 'camera', name=side + '_wrist',
                      pos='0 -.07 .06', xyaxes='-1 0 0 0 .789 -.614', fovy='70')


def boom_wrist_cameras(world):
    # Outboard boom (tool -Y right, +Y left), looking down past the fingertips toward the midline.
    for side, k in (('left', -1), ('right', 1)):
        ET.SubElement(world.find(f".//body[@name='{side}_right_hand']"), 'camera', name=side + '_wrist',
                      pos=f'-.05 {-.20 * k} .06', xyaxes=f'{.979 * k} 0 {-.204 * k} .146 {-.70 * k} .699', fovy='75')


def scene(wrist_cameras='hand', bottle_pose=None):
    root = robot_scene()
    root.set('model', 'dual_arm_free_bottle')
    world = root.find('worldbody')
    ET.SubElement(root.find('visual'), 'headlight', ambient='.5 .5 .5', diffuse='.5 .5 .5', specular='.15 .15 .15')
    ET.SubElement(root.find('asset'), 'texture', type='skybox', builtin='gradient', rgb1='.22 .27 .33', rgb2='.08 .11 .15', width='256', height='1536')
    bottle = ET.fromstring(scene_xml())
    # Preserve the small latch's timestep, exact inertias, joint limits and all
    # 294 contact pairs. Named defaults isolate it from robot armature/friction.
    root.find('option').set('timestep', bottle.find('option').get('timestep'))
    root.find('option').set('cone', 'elliptic')
    defaults = ET.SubElement(root.find('default'), 'default', {'class': 'bottle'})
    for element in bottle.find('default'):
        defaults.append(deepcopy(element))
    defaults.find('joint').set('armature', '0')
    # External robot/table contact uses bit 1, bottle bit 2. Internal bottle
    # contact still uses its original explicit pair set (no shell self-contact).
    for geom in root.iter('geom'):
        if geom.get('contype') != '0': geom.set('conaffinity', '3')
    for item in bottle.find('asset'):
        if item.tag in ('mesh', 'material') and item.get('name') != 'floor':
            root.find('asset').append(deepcopy(item))
    body = deepcopy(bottle.find("worldbody/body[@name='bottle']"))
    body.set('pos', '0 0 .8005'); body.set('childclass', 'bottle')
    if bottle_pose is not None:
        x, y, yaw_deg = bottle_pose
        # The compiler uses angle="radian", so the documented degrees must be converted.
        body.set('pos', f'{x} {y} .8005'); body.set('euler', f'0 0 {math.radians(yaw_deg)}')
    ET.SubElement(body, 'freejoint', name='bottle_free')
    ET.SubElement(body, 'inertial', pos='0 0 .11', mass='.35', diaginertia='.00153 .00153 .000227')
    # Primitive outer shells and convex mechanism pieces are physical; cosmetic
    # rings stay visual to avoid filling their holes with a convex hull.
    solid = ('base', 'bottle_wall', 'shoulder', 'collar', 'lid_top', 'button_face',
             'rocker_spine', 'rocker_hook', 'hinge_base')
    for geom in body.iter('geom'):
        name = geom.get('name', '')
        if name in solid or name.startswith(('cap_slot_', 'catch_arc_')):
            geom.set('contype', '2'); geom.set('conaffinity', '1')
            geom.set('friction', '1 .01 .001')
    # The lid wall is a hollow mesh; a ring of small wall pieces gives robot contact.
    lid = body.find("body[@name='lid']")
    for i in range(32):
        angle = 2 * math.pi * i / 32
        ET.SubElement(lid, 'geom', name=f'lid_wall_contact_{i}', type='box',
                      pos=f'{.033*math.cos(angle)} {-.031+.033*math.sin(angle)} .008',
                      euler=f'0 0 {angle}', size='.002 .00325 .009',
                      contype='2', conaffinity='1', mass='0', rgba='0 0 0 0', friction='1 .01 .001')
    world.append(body)
    for pair in bottle.find('contact'):
        pair = deepcopy(pair); pair.set('class', 'bottle'); root.find('contact').append(pair)
    # Bottle actuators are deliberately NOT copied. No welds/equalities.
    for side in ('left', 'right'):
        hand = world.find(f".//body[@name='{side}_right_hand']")
        hand.find(f"site[@name='{side}_tip']").set('pos', '0 0 .142')
        for finger, sign in [('a', -1), ('b', 1)]:
            name = f'{side}_finger_{finger}'
            g = world.find(f".//geom[@name='{name}_pad']")
            g.set('pos', f'{sign*.004} 0 .130'); g.set('size', '.004 .004 .012')
            g.set('friction', '1.5 .01 .001'); g.set('solref', '.001 1')
            b = world.find(f".//body[@name='{name}']")
            ET.SubElement(b, 'geom', name=name + '_shaft', type='box', pos=f'{sign*.004} 0 .09',
                          size='.004 .005 .028', mass='.025', rgba='.40 .44 .48 1', conaffinity='3')
            touch = b.find(f"site[@name='{name}_touch']")
            touch.set('pos', f'{sign*.001} 0 .130'); touch.set('size', '.003 .005 .013')
    {'hand': hand_wrist_cameras, 'boom': boom_wrist_cameras}[wrist_cameras](world)
    # Opposed workspaces leave access to the front button and bottle side.
    world.find("body[@name='left_base']").set('pos', '-.40 -.30 .8')
    right = world.find("body[@name='right_base']")
    right.set('pos', '.48 .27 .8'); right.set('quat', '0 0 0 1')
    return ET.tostring(root, encoding='unicode')
