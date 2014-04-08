use std::cell::Cell;

static EPSILON: f32 = 0.00001;

type Color = Vector;
type Light = Vector;
type Point = Vector;

trait Intersect{
  fn intersect(&self, ray: &Ray) -> Option<f32>;
  fn normalAt(&self, p: &Vector) -> Vector;
}

#[deriving(Eq,Show,Clone)]
struct Vector {
  x: f32,
  y: f32,
  z: f32,
}

impl Vector {
  fn new(x:f32, y:f32, z:f32) -> Vector {
    Vector{x:x, y:y, z:z}
  }

  fn magnitude(&self) -> f32 {
    self.dot(self).sqrt()
  }

  fn scale(&self, c: f32) -> Vector {
    Vector{
      x: self.x * c,
      y: self.y * c,
      z: self.z * c,
    }
  }

  fn dot(&self, other: &Vector) -> f32 {
    self.x * other.x + self.y * other.y + self.z * other.z
  }

  fn cross(&self, other: &Vector) -> Vector {
    Vector{
      x: self.y * other.z - self.z * other.y,
      y: self.z * other.x - self.x * other.z,
      z: self.x * other.y - self.y * other.x
    }
  }

  fn normalized(&self) -> Vector {
    self.scale(1.0 / self.magnitude())
  }

  fn reflectThrough(self, normal: &Vector) -> Vector {
    self - (normal.scale(self.dot(normal))).scale(2.0)
  }
}

impl Add<Vector,Vector> for Vector {
  fn add(&self, rhs: &Vector) -> Vector {
    Vector{
      x: self.x + rhs.x,
      y: self.y + rhs.y,
      z: self.z + rhs.z,
    }
  }
}

impl Sub<Vector,Vector> for Vector {
  fn sub(&self, rhs: &Vector) -> Vector {
    *self + rhs.scale(-1.0)
  }
}

static VZERO: Vector = Vector{x:0.0,y:0.0,z:0.0};
static VUP: Vector = Vector{x:0.0,y:1.0,z:0.0};

#[test]
fn vectorTest() {
  assert!(VRIGHT.reflectThrough(&VUP) == VRIGHT)
  assert!(Vector::new(-1.0,-1.0,0.0).reflectThrough(&VUP) == Vector::new(-1.0,1.0,0.0))
}

struct Sphere {
  centre: Vector,
  radius: f32,
}

impl Sphere {
  fn new(centre: Vector, radius: f32) -> Sphere {
    Sphere{centre:centre,radius:radius}
  }
}

impl Intersect for Sphere {
  fn intersect(&self, ray: &Ray) -> Option<f32> {
    let cp = self.centre - ray.point;
    let v = cp.dot(&ray.vector);
    let discriminant = (self.radius * self.radius) - (cp.dot(&cp) - v*v);
    if discriminant < 0.0 {
      None
    } else {
      Some(v - discriminant.sqrt())
    }
  }
  fn normalAt(&self, p: &Vector) -> Vector {
    (p - self.centre).normalized()
  }
}

#[deriving(Show,Eq)]
struct Halfspace {
  point: Vector,
  normal: Vector,
}

impl Halfspace {
  fn new(point: Vector, normal: Vector) -> Halfspace {
    Halfspace{ point:point, normal: normal.normalized() }
  }
}

impl Intersect for Halfspace {
  fn intersect(&self, ray: &Ray) -> Option<f32> {
    let v = ray.vector.dot(&self.normal);
    if v != 0.0 {
      //Some( 1.0 / -v )
      Some( (self.point - ray.point).dot(&self.normal) / v)
    } else {
      None
    }
  }
  fn normalAt(&self, _p: &Vector) -> Vector {
    self.normal
  }
}

#[deriving(Show,Eq)]
struct Ray {
  point: Vector,
  vector: Vector,
}

impl Ray {
  fn new(point: Vector, vector: Vector) -> Ray {
    Ray{point:point, vector:vector.normalized()}
  }

  fn pointAtTime(&self, t: f32) -> Vector {
    self.point + self.vector.scale(t)
  }
}

struct PpmCanvas<'a> {
  width: uint,
  height: uint,
  filenameBase: &'a str,
  bytes: Vec<u8>,
}

impl<'a> PpmCanvas<'a> {
  fn new(width: uint, height: uint, filenameBase: &'a str) -> PpmCanvas<'a> {
    PpmCanvas {
      width: width,
      height: height,
      filenameBase: filenameBase,
      bytes: Vec::from_fn(width * height * 3,|_|0)
    }
  }

  fn plot(&mut self, x: uint, y: uint, r: f32, g: f32, b: f32) {
    use std::cmp::{max,min};
    //let i = ((self.height - y - 1) * self.width + x) * 3;
    let i = (y * self.width + x) * 3;
    *self.bytes.get_mut(i  ) = max(0, min(255, (r * 255.0) as u8));
    *self.bytes.get_mut(i+1) = max(0, min(255, (g * 255.0) as u8));
    *self.bytes.get_mut(i+2) = max(0, min(255, (b * 255.0) as u8));
  }

  fn save(&self) {
    use std::io::{Writer,File,Open,Write};
    use std::path::Path;
    let mut writer = File::open_mode(&Path::new(self.filenameBase), Open, Write).unwrap();
    let _ = writer.write_str( format!("P6\n{} {}\n255\n", self.width, self.height) );
    let _ = writer.write(self.bytes.as_slice());
  }
}

struct Scene {
  objects: Vec<(~Intersect,~Surface)>,
  lights: Vec<Light>,
  position: Vector,
  lookingAt: Vector,
  fieldOfView: f32,
  recursionDepth: Cell<uint>,
  recursionLimit: uint,
}

impl Scene {
  fn new() -> Scene {
    Scene {
      objects: Vec::new(),
      lights: Vec::new(),
      position: Vector::new(0.0, 1.8, 10.0),
      lookingAt: VZERO,
      fieldOfView: 45.0,
      recursionDepth: Cell::new(0),
      recursionLimit: 3
    }
  }

  fn lookAt(&mut self, point: Vector) {
    self.lookingAt = point;
  }

  fn addObject<T:'static + Intersect,S:'static + Surface>(&mut self, object: ~T, s: ~S) {
    self.objects.push((object as ~Intersect,s as ~Surface));
  }

  fn addLight(&mut self, light: Light) {
    self.lights.push(light);
  }

  fn render(&mut self, canvas: &mut PpmCanvas) {
    use std::f32::consts::PI;
    let fovRadians = PI * (self.fieldOfView / 2.0) / 180.0;
    let halfWidth = fovRadians.tan();
    let halfHeight = halfWidth;
    let width = halfWidth * 2.0;
    let height = halfHeight * 2.0;
    let pixelWidth = width / (canvas.width as f32 - 1.0);
    let pixelHeight = height / (canvas.height as f32 - 1.0);

    let eye = Ray::new(self.position, self.lookingAt - self.position);
    let vpRight = eye.vector.cross(&VUP).normalized();
    let vpUp = vpRight.cross(&eye.vector).normalized();

    for y in range(0, canvas.height) {
      for x in range(0, canvas.width) {
        let xcomp = vpRight.scale(x as f32 * pixelWidth - halfWidth);
        let ycomp = vpUp.scale(y as f32 * pixelHeight - halfHeight);
        let ray = Ray::new(eye.point, eye.vector + xcomp + ycomp);
        let color = self.rayColour(&ray);
        canvas.plot(x,y,color.x,color.y,color.z);
      }
    }
    canvas.save();
  }

  fn rayColour(&self, ray: &Ray) -> Color {
    use std::f32;
    if self.recursionDepth.get() > self.recursionLimit {
      VZERO
    } else {
      self.recursionDepth.set(self.recursionDepth.get() + 1);

      let mut minT = f32::MAX_VALUE as f32;
      let mut minO = None;
      for &(ref obj, ref s) in self.objects.iter() {
        match obj.intersect(ray) {
          Some(t) if (t > -EPSILON && t < minT) => {
            minT = t;
            minO = Some((obj,t,s))
          },
          _ => {}
        }
      }
      let result = match minO {
        None => VZERO,
        Some((o, t, s)) => {
          let p = ray.pointAtTime(t);
          s.colourAt(self, ray, &p, &o.normalAt(&p))
        }
      };
      self.recursionDepth.set(self.recursionDepth.get() - 1);
      result
    }
  }

  fn lightIsVisible(&self, l: &Light, p: &Vector) -> bool {
    let ray = Ray::new(*p,l-(*p));
    let length = (*l-*p).magnitude();
    for &(ref o, _) in self.objects.iter() {
      let t = o.intersect(&ray);
      match t {
        Some(f) if f > EPSILON && f < length - EPSILON => {
          return false;
        },
        _ => {}
      }
      //return t.map_or(true, |t| t <= EPSILON);
    }
    true
  }

  fn visibleLights(&self, p: &Vector) -> Vec<Light> {
    self.lights.iter()
      .filter(|&l| self.lightIsVisible(l,p))
      .map(|l|*l).collect()
  }
}

fn addColours(a: Color, scale: f32, b: Color) -> Color {
  Vector{
    x: a.x + scale * b.x,
    y: a.y + scale * b.y,
    z: a.z + scale * b.z
  }
}

struct SimpleSurface {
  baseColour: Color,
  specularCoefficient: f32,
  lambertCoefficient: f32,
}

trait Surface {
  fn baseColourAt(&self, p: &Vector) -> Color;
  fn colourAt(&self, scene: &Scene, ray: &Ray, p: &Vector , normal: &Vector) -> Color {
    use std::f32::abs;
    let b = self.baseColourAt(p);
    let mut c = Vector::new(0.0, 0.0, 0.0);
    if self.getSpecular() > 0.0 {
      let reflectedRay = Ray::new(*p, ray.vector.reflectThrough(normal));
      let reflectedColour = scene.rayColour(&reflectedRay);
      c = addColours(c, self.getSpecular(), reflectedColour)
    }
    if self.getLambert() > 0.0 {
      let mut lambertAmount = 0.0;
      for lightPoint in scene.visibleLights(p).iter() {
        let d = *p - *lightPoint;
        let dLength = d.magnitude();
        let contribution = abs(d.dot(normal) / (dLength * dLength));
        lambertAmount = lambertAmount + contribution;
      }
      c = addColours(c, self.getLambert() * lambertAmount, b)
    }
    if self.getAmbient() > 0.0 {
      c = addColours(c, self.getAmbient(), b)
    }
    c
  }
  fn getSpecular(&self) -> f32;
  fn getLambert(&self) -> f32;
  fn getAmbient(&self) -> f32 {
    1.0 - self.getSpecular() - self.getLambert()
  }
}

impl SimpleSurface {
  fn new(baseColour: Color) -> SimpleSurface {
    SimpleSurface::advancedNew(baseColour, 0.3, 0.6)
  }
  fn advancedNew(baseColour: Color, specularCoefficient: f32, lambertCoefficient: f32) -> SimpleSurface {
    SimpleSurface {
      baseColour: baseColour,
      specularCoefficient: specularCoefficient,
      lambertCoefficient: lambertCoefficient,
    }
  }
}

impl Surface for SimpleSurface {
  fn baseColourAt(&self, _: &Vector) -> Color {
    self.baseColour
  }
  fn getSpecular(&self) -> f32 {
    self.specularCoefficient
  }
  fn getLambert(&self) -> f32 {
    self.lambertCoefficient
  }
}

struct CheckerboardSurface {
  baseColour: Color,
  otherColour: Color,
  checkSize: f32,
  specularCoefficient: f32,
  lambertCoefficient: f32,
}

impl CheckerboardSurface {
  fn new(baseColour: Color, otherColour: Color, checkSize: f32) -> CheckerboardSurface {
    CheckerboardSurface::advancedNew(baseColour, otherColour, checkSize, 0.3, 0.6)
  }
  fn advancedNew(baseColour: Color, otherColour: Color, checkSize: f32, specularCoefficient: f32,
        lambertCoefficient: f32) -> CheckerboardSurface {
    CheckerboardSurface {
      baseColour: baseColour,
      otherColour: otherColour,
      checkSize: checkSize,
      specularCoefficient: specularCoefficient,
      lambertCoefficient: lambertCoefficient,
    }
  }

}

impl Surface for CheckerboardSurface {
  fn baseColourAt(&self, p: &Vector) -> Color {
    let v = (p - VZERO).scale(1.0 / self.checkSize);
    fn f(x:f32) -> uint {
      use std::num::{abs,Round};
      (abs(x) + 0.5).floor() as uint
    }
    if (f(v.x) + f(v.y) + f(v.z)) % 2 == 1 {
      self.otherColour
    } else {
      self.baseColour
    }
  }
  fn getSpecular(&self) -> f32 {
    self.specularCoefficient
  }
  fn getLambert(&self) -> f32 {
    self.lambertCoefficient
  }
}

fn coolScene() -> Scene {
  use std::f32::consts::PI;
  let mut s = Scene::new();
  for i in range(0,10) {
    let i = i as f32;
    let theta = i * (i+5.0) * PI / 100.0 + 0.3;
    let center = Vector::new(0.0 - 4.0 * theta.sin(), 1.5 - i / 2.0, 0.0 - 4.0 * theta.cos());
    let form = Sphere::new(center, 0.3 + i * 0.1);
    let surface = SimpleSurface::new(Vector::new(i / 6.0, 1.0 - i/6.0, 0.5));
    s.addObject(box form, box surface);
  }
  s.addObject(
    box Sphere::new(VZERO, 2.0),
    box SimpleSurface::new(Vector::new(1.0,1.0,1.0))
  );
  s.addObject(
    box Halfspace::new(Vector::new(0.0,4.0,0.0),Vector::new(0.0,1.0,0.0)),
    box CheckerboardSurface::new(Vector::new(1.0,1.0,1.0),VZERO,1.0)
  );
  s.addObject(
    box Halfspace::new(Vector::new(0.0,-4.0,0.0),Vector::new(0.0,1.0,0.0)),
    box SimpleSurface::new(Vector::new(0.9,1.0,1.0))
  );
  s.addObject(
    box Halfspace::new(Vector::new(6.0,0.0,0.0),Vector::new(1.0,0.0,0.0)),
    box SimpleSurface::new(Vector::new(1.0,0.9,1.0))
  );
  s.addObject(
    box Halfspace::new(Vector::new(-6.0,0.0,0.0),Vector::new(1.0,0.0,0.0)),
    box SimpleSurface::new(Vector::new(1.0,1.0,0.9))
  );
  s.addObject(
    box Halfspace::new(Vector::new(0.0,0.0,6.0),Vector::new(0.0,0.0,1.0)),
    box SimpleSurface::new(Vector::new(0.9,0.9,1.0))
  );
  s.addLight(
    Vector::new(0.0,-3.0,0.0)
  );
  s.addLight(
    Vector::new(3.0,3.0,0.0)
  );
  s.addLight(
    Vector::new(-3.0,3.0,0.0)
  );
  s.position = Vector::new(0.0,0.0,-15.0);
  s.lookingAt = Vector::new(0.0,0.0,0.0);
  s.fieldOfView = 45.0;
  s.recursionLimit = 5;
  s
}

fn normalScene() -> Scene {
  let mut s = Scene::new();
  s.addLight(Vector::new(30.0, 30.0, 10.0));
  s.addLight(Vector::new(-10.0, 100.0, 30.0));
  s.lookAt(Vector::new(0.0, 3.0, 0.0));
  s.addObject(
    box Sphere::new(Vector::new(1.0,3.0,-10.0), 2.0),
    box SimpleSurface::new(Vector::new(1.0, 1.0, 0.0))
  );
  for x in range(0,6) {
    let y = x as f32;
    s.addObject(
      box Sphere::new(Vector::new(-3.0-y*0.4, 2.3, -5.0), 0.4),
      box SimpleSurface::new(Vector::new(y/6.0, 1.0-y/6.0, 0.5))
    );
  }
  s.addObject(
    box Halfspace::new(VZERO, VUP),
    box CheckerboardSurface::new(Vector::new(1.0,1.0,1.0), VZERO, 1.0)
  );
  s
}

fn main() {
  let mut canvas = PpmCanvas::new(800,800,"test_raytrace_rust.ppm");

  let mut s = coolScene();
  s.render(&mut canvas);
  /*for a in range(-2,2) {
    for b in range(-2,2) {
      for c in range(-2,2) {
        let name = format!("test_raytrace_rust_{}_{}_{}.ppm",a,b,c);
        let mut canv = PpmCanvas::new(800,600,name);
        s.position = Vector::new(a as f32, b as f32, c as f32);
        s.render(&mut canv);
      }
    }
  }*/
}
