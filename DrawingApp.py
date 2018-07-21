import pygame, random
import pygame.surfarray as surfarray
import Mnist

def main():
    sess = Mnist.train()
    screen = pygame.display.set_mode((28,28)) #THIS IS A SURFACE OBJECT

    draw_on = False
    last_pos = (0, 0)
    color = (255, 128, 0)
    radius = 1

    def roundline(srf, color, start, end, radius=1):
        dx = end[0]-start[0]
        dy = end[1]-start[1]
        distance = max(abs(dx), abs(dy))
        for i in range(distance):
            x = int( start[0]+float(i)/distance*dx)
            y = int( start[1]+float(i)/distance*dy)
            pygame.draw.circle(srf, color, (x, y), radius)

    try:
        while True:
            e = pygame.event.wait()
            if e.type == pygame.QUIT:
                raise StopIteration
            if e.type == pygame.MOUSEBUTTONDOWN:
                color = (255, 255, 255)
                pygame.draw.circle(screen, color, e.pos, radius)
                draw_on = True
            if e.type == pygame.MOUSEBUTTONUP:
                draw_on = False
            if e.type == pygame.MOUSEMOTION:
                if draw_on:
                    pygame.draw.circle(screen, color, e.pos, radius)
                    roundline(screen, color, e.pos, last_pos,  radius)
                last_pos = e.pos
            pygame.display.flip()

            new_array = []
            arr2d = surfarray.array2d(screen)
            for arr in arr2d:
                for integer in arr:
                    #if(integer == -256):
                    #    new_array.append(256)
                    #else:
                    new_array.append(integer)

            print(Mnist.Mnist(new_array, sess))

    except StopIteration:
        pass

    pygame.quit()    

if __name__ == '__main__':
    main()
